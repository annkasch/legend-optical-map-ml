"""Training-time reading and shuffling of a photon store (photon_store.py).

Batches are float32 tensors [M, 7]: x, y, z (m), px, py, pz (MeV) and the label bits
(``labels(batch)``). Normalization is left to the consumer (``normalization()``), so the
data on disk and in transfer stay raw.

One pass p (seed ``hash(seed, p)``):

1. every shard is cut into blocks of ``block_photons`` with a random boundary offset per
   shard and pass, so photons are regrouped every pass;
2. all blocks are permuted (streaming) and dealt round-robin to the ranks, or, with
   ``in_memory``, each rank permutes the blocks of the shards it owns;
3. consecutive blocks form a buffer (≈ ``buffer_photons``, so it mixes about
   ``buffer_photons / block_photons`` shards); the buffer is permuted on the target device;
4. the permuted buffers form one stream, cut into batches of ``batch_size``.

Every photon of the rank's blocks appears exactly once per pass (``drop_last=False``).
With several ranks every rank yields the same number of batches (``n_batches``); the few
photons beyond that are skipped in this pass only (block offsets differ every pass).
No global permutation index exists: a pass is a list of blocks.

Reading: background threads read whole blocks with ``os.preadv`` straight into a ring of
host buffers (pinned for CUDA), hint the kernel with ``posix_fadvise``, and the consumer
copies a filled buffer to the device on a side stream, so reading, transfer and training
overlap. ``in_memory`` keeps the rank's shards in RAM after the first read; small blocks
(default 4096 photons) then give an almost complete shuffle every pass without I/O.

A pass is deterministic for a given (seed, pass, rank, world size, device type), so
``batches(start_batch=k)`` resumes after a checkpoint without replaying earlier batches.
"""

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legend_optical_map_ml.photon_store import RECORD, load_manifest, normalization, shards

WIDTH = 7  # float32 values per record
STREAM_BLOCK = 2**17  # 131072 photons = 3.7 MB per sequential read
MEMORY_BLOCK = 2**12
BUFFER_PHOTONS = 2**20  # ≈ 29 MB per buffer


def labels(batch):
    """Labels [M] (int32, 1 = detected) of a batch [M, 7] (torch or numpy)."""
    column = batch[:, 6]
    if isinstance(batch, np.ndarray):
        return np.ascontiguousarray(column).view(np.int32)
    return column.contiguous().view(_torch().int32)


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class Block:
    shard: int  # position in PhotonStream.shards
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


class PhotonStream:
    def __init__(
        self,
        store: str | Path,
        split: str = "train",
        *,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        block_photons: int | None = None,
        buffer_photons: int = BUFFER_PHOTONS,
        in_memory: bool = False,
        prefetch: int = 3,
        threads: int = 4,
    ):
        if not 0 <= rank < world_size:
            raise ValueError("rank must lie in [0, world_size)")
        self.store = Path(store)
        self.manifest = load_manifest(self.store)
        self.split = split
        self.shards = shards(self.manifest, split)
        if not self.shards:
            raise ValueError(f"No {split} shards in {self.store}")
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.in_memory = in_memory
        self.block_photons = block_photons or (MEMORY_BLOCK if in_memory else STREAM_BLOCK)
        self.buffer_blocks = max(1, buffer_photons // self.block_photons)
        self.prefetch, self.threads = max(1, prefetch), max(1, threads)
        self._memory: dict[int, np.ndarray] = {}
        self._owned = self._owned_shards() if in_memory else range(len(self.shards))

    # -- sizes and metadata ---------------------------------------------------------------------

    @property
    def n_photons(self) -> int:
        """Photons of the split (all ranks)."""
        return sum(s["n_photons"] for s in self.shards)

    @property
    def n_detected(self) -> int:
        """Detected photons of the split (all ranks)."""
        return sum(s["n_detected"] for s in self.shards)

    # RESUM_FLEX EventSource protocol (core.event_stream): events = photons, positive = detected.
    @property
    def n_events(self) -> int:
        return self.n_photons

    @property
    def n_positive(self) -> int:
        return self.n_detected

    def normalization(self, method: str = "standard") -> dict:
        """Normalization from the store's train shards (identical for every split)."""
        return normalization(self.manifest, method)

    def _owned_shards(self) -> list[int]:
        """In-memory mode: shards of this rank, balanced by photons (fixed for all passes)."""
        loads = [0] * self.world_size
        owner = {}
        order = sorted(range(len(self.shards)), key=lambda i: -self.shards[i]["n_photons"])
        for i in order:
            r = int(np.argmin(loads))
            owner[i] = r
            loads[r] += self.shards[i]["n_photons"]
        return [i for i in range(len(self.shards)) if owner[i] == self.rank]

    # -- pass plan ------------------------------------------------------------------------------

    def _rng(self, *words) -> np.random.Generator:
        return np.random.default_rng([self.seed, *words])

    def _blocks(self, pass_index: int, shard_indices) -> list[Block]:
        rng = self._rng(pass_index, 0)
        size = self.block_photons
        blocks = []
        for i in range(len(self.shards)):  # same draws for every rank and shard subset
            n = self.shards[i]["n_photons"]
            offset = int(rng.integers(0, size)) if n > size else 0
            if i not in shard_indices:
                continue
            edges = [0, *range(offset or size, n, size), n]
            pairs = zip(edges[:-1], edges[1:], strict=True)
            blocks.extend(Block(i, a, b) for a, b in pairs if b > a)
        return blocks

    def rank_blocks(self, pass_index: int, rank: int | None = None) -> list[Block]:
        """Blocks of one rank in reading order for one pass."""
        rank = self.rank if rank is None else rank
        if self.in_memory:
            owned = set(self._owned) if rank == self.rank else set(self._owned_of(rank))
            blocks = self._blocks(pass_index, owned)
            order = self._rng(pass_index, 1, rank).permutation(len(blocks))
            return [blocks[i] for i in order]
        blocks = self._blocks(pass_index, set(range(len(self.shards))))
        order = self._rng(pass_index, 1).permutation(len(blocks))
        return [blocks[i] for i in order[rank :: self.world_size]]

    def _owned_of(self, rank: int) -> list[int]:
        saved, self.rank = self.rank, rank
        try:
            return self._owned_shards()
        finally:
            self.rank = saved

    def buffers(self, pass_index: int) -> list[list[Block]]:
        blocks = self.rank_blocks(pass_index)
        k = self.buffer_blocks
        return [blocks[i : i + k] for i in range(0, len(blocks), k)]

    def rank_photons(self, pass_index: int) -> list[int]:
        ranks = range(self.world_size)
        return [sum(b.size for b in self.rank_blocks(pass_index, r)) for r in ranks]

    def n_batches(self, batch_size: int, pass_index: int, *, drop_last: bool = False) -> int:
        """Batches per rank in this pass; equal for all ranks when world_size > 1."""
        photons = self.rank_photons(pass_index)
        if self.world_size > 1 or drop_last:
            return min(photons) // batch_size
        return -(-photons[0] // batch_size)

    # -- reading --------------------------------------------------------------------------------

    def _path(self, shard: int) -> Path:
        return self.store / self.shards[shard]["records"]

    def _load_owned(self, pool: ThreadPoolExecutor) -> None:
        """In-memory mode: read the rank's shards that are not in RAM yet (in parallel)."""
        missing = [i for i in self._owned if i not in self._memory]

        def read(i):
            array = np.empty((self.shards[i]["n_photons"], WIDTH), dtype=np.float32)
            self._read_into(i, 0, len(array), array)
            return i, array

        for i, array in pool.map(read, missing):
            self._memory[i] = array

    def _read_into(self, shard: int, start: int, stop: int, out: np.ndarray) -> None:
        """Read records [start, stop) of a shard into ``out`` [stop - start, 7] float32."""
        size = RECORD.itemsize
        view = memoryview(out.reshape(-1).view(np.uint8))
        fd = os.open(self._path(shard), os.O_RDONLY)
        try:
            offset, done = start * size, 0
            total = (stop - start) * size
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, offset, total, os.POSIX_FADV_SEQUENTIAL)
            while done < total:
                count = os.preadv(fd, [view[done:total]], offset + done)
                if count <= 0:
                    raise OSError(f"Short read in {self._path(shard)}")
                done += count
        finally:
            os.close(fd)

    def _fill(self, blocks: list[Block], out: np.ndarray, pool: ThreadPoolExecutor) -> int:
        """Read a buffer's blocks into ``out`` in order; returns the number of photons."""
        jobs, position = [], 0
        for block in blocks:
            target = out[position : position + block.size]
            if self.in_memory:
                source = self._memory[block.shard][block.start : block.stop]
                jobs.append(pool.submit(np.copyto, target, source))
            else:
                jobs.append(
                    pool.submit(self._read_into, block.shard, block.start, block.stop, target)
                )
            position += block.size
        for job in jobs:
            job.result()
        return position

    def load(self) -> np.ndarray:
        """All photons of the split as float32 [N, 7] in shard order (e.g. validation)."""
        out = np.empty((self.n_photons, WIDTH), dtype=np.float32)
        position = 0
        for i, shard in enumerate(self.shards):
            n = shard["n_photons"]
            self._read_into(i, 0, n, out[position : position + n])
            position += n
        return out

    # -- batches --------------------------------------------------------------------------------

    def batches(
        self,
        batch_size: int,
        pass_index: int = 0,
        *,
        start_batch: int = 0,
        drop_last: bool = False,
        device="cpu",
    ):
        """Yield batches [batch_size, 7] float32 on ``device`` for one pass of this rank."""
        torch = _torch()
        device = torch.device(device)
        cuda = device.type == "cuda"
        limit = self.n_batches(batch_size, pass_index, drop_last=drop_last)
        if start_batch >= limit:
            return
        plan = self.buffers(pass_index)
        # Skip whole buffers before the resume point without reading them.
        skip = start_batch * batch_size
        first = 0
        while first < len(plan) and skip >= sum(b.size for b in plan[first]):
            skip -= sum(b.size for b in plan[first])
            first += 1
        capacity = max(sum(b.size for b in buffer) for buffer in plan)
        slots = queue.Queue()
        for _ in range(self.prefetch + 1):
            host = torch.empty((capacity, WIDTH), dtype=torch.float32, pin_memory=cuda)
            slots.put(host)
        ready = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def produce():
            try:
                with ThreadPoolExecutor(self.threads) as pool:
                    if self.in_memory:
                        self._load_owned(pool)
                    for index in range(first, len(plan)):
                        host = slots.get()
                        if stop.is_set():
                            return
                        n = self._fill(plan[index], host.numpy(), pool)
                        ready.put((index, host, n))
                ready.put(None)
            except BaseException as exc:  # surfaced in the consumer
                ready.put(exc)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        copy_stream = torch.cuda.Stream(device) if cuda else None
        pending, emitted = None, start_batch
        in_flight = []  # (event, host) of copies not yet known to be complete
        try:
            while emitted < limit:
                item = ready.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                index, host, n = item
                if cuda:
                    with torch.cuda.stream(copy_stream):
                        data = host[:n].to(device, non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(copy_stream)
                    torch.cuda.current_stream(device).wait_event(event)
                    in_flight.append((event, host))
                    generator = torch.Generator(device=device)
                else:
                    data = host[:n].clone()
                    slots.put(host)
                    generator = torch.Generator()
                generator.manual_seed(_buffer_seed(self.seed, pass_index, self.rank, index))
                data = data[torch.randperm(n, generator=generator, device=device)]
                if index == first and skip:
                    data = data[skip:]
                pending = data if pending is None else torch.cat([pending, data])
                while len(pending) >= batch_size and emitted < limit:
                    yield pending[:batch_size]
                    pending = pending[batch_size:]
                    emitted += 1
                while in_flight and in_flight[0][0].query():
                    slots.put(in_flight.pop(0)[1])
                if cuda and len(in_flight) > self.prefetch:
                    event, host = in_flight.pop(0)
                    event.synchronize()
                    slots.put(host)
            if pending is not None and len(pending) and emitted < limit:
                yield pending
        finally:
            stop.set()
            for event, host in in_flight:
                event.synchronize()
                slots.put(host)
            slots.put(None)
            while thread.is_alive():
                try:
                    ready.get(timeout=0.1)
                except queue.Empty:
                    pass
                slots.put(None)
            thread.join()


def _buffer_seed(seed: int, pass_index: int, rank: int, index: int) -> int:
    """63-bit seed of one buffer's permutation."""
    state = np.random.SeedSequence([seed, pass_index, 2, rank, index]).generate_state(1, np.uint64)
    return int(state[0]) >> 1
