import torch
from time import time
import gc


class Data_Prefetcher:
    def __init__(self, loader, prefetch_factor=3):
        # torch.manual_seed(config.seed)
        # torch.cuda.manual_seed(config.seed)

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.prefetch_factor = prefetch_factor
        self.prefetched_data = []
        self.preload()

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            for _ in range(self.prefetch_factor):
                input, target = next(self.loader)
                with torch.cuda.stream(self.stream):
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                self.prefetched_data.append((input, target))
                # print(f'prefetcher preloaded {len(self.prefetched_data)}')

        except StopIteration:
            self.prefetched_data.append((None, None))

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if len(self.prefetched_data) == 0:
            return None, None

        input_, target = self.prefetched_data.pop(0)
        # print(f'prefetcher next called: prefetcher {len(self.prefetched_data)}')
        self.preload()  # start preloading next batches
        return input_, target

        # torch.cuda.current_stream().wait_stream(self.stream)
        # input_ = self.next_input
        # target = self.next_target
        # self.preload()
        # return input_, target


def collate_fn_vec_high_dim(batch, config, shuffling_method=0, debug=False):
    """
    batch: [
            (seqC, theta, probR),
            (seqC, theta, probR),
            ...
            (seqC, theta, probR),
            ]
            seqC: (DM, S, L), theta: (4,), probR: (DM, S, 1)

    """

    C = config["dataset"]["num_probR_sample"]
    B = len(batch)

    if debug:
        start_time_0 = time.time()

    # # Preallocate tensors
    # seqC_batch = torch.empty((B, *batch[0][0].shape))
    # theta_batch = torch.empty((B, *batch[0][1].shape))
    # probR_batch = torch.empty((B, *batch[0][2].shape))

    # # Fill tensors with data from the batch
    # for i, (seqC, theta, probR) in enumerate(batch):
    #     seqC_batch[i] = seqC
    #     theta_batch[i] = theta
    #     probR_batch[i] = probR

    # seqC, theta, probR = zip(*batch)

    # seqC_batch = torch.stack(seqC)
    # theta_batch = torch.stack(theta)
    # probR_batch = torch.stack(probR)

    seqC_batch, theta_batch, probR_batch = map(torch.stack, zip(*batch))

    if debug:
        print(
            f"collate_fn_vec_high_dim: dataloading {(time.time() - start_time_0)*1000:.2f} ms"
        )

    # del batch, seqC, theta, probR

    # Repeat seqC and theta C times along a new dimension
    seqC_batch = seqC_batch.repeat_interleave(
        C, dim=0
    )  # (C*B, DM, S, 15) first C samples are the same, from the first batch
    theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)

    # Repeat probR C times along a new dimension and sample from Bernoulli distribution
    probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, DM, S, 1)

    # bernouli sampling choice, and concatenate x_seqC and x_choice
    x_batch = torch.cat(
        [seqC_batch, probR_batch.bernoulli_()], dim=-1
    )  # (C*B, DM, S, 16)
    del probR_batch, seqC_batch
    gc.collect()

    if debug:
        print(
            f"\ncollate_fn_vec: get x_batch {(time.time() - start_time_0)*1000:.2f} ms"
        )
        start_time = time.time()

    # Shuffle x along the 3rd axis S, using advanced indexing
    BC, DM, S, _ = x_batch.shape
    permutations = torch.stack(
        [torch.stack([torch.randperm(S) for _ in range(DM)]) for _ in range(BC)]
    )  # (BC, DM, S)

    # for i in range(BC):
    #     for j in range(DM):
    #         x_batch[i, j] = x_batch[i, j, permutations[i, j]]
    # Shuffle the batched dataset

    indices = torch.randperm(BC)
    BC_range = indices[:, None, None]
    DM_range = torch.arange(DM)[None, :, None]

    # if debug:
    #     print(f"collate_fn_vec: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")
    #     start_time = time.time()

    return (
        x_batch[BC_range, DM_range, permutations],
        theta_batch[indices],
    )  # ! check the output shape and shuffling result and logic

    # return x_batch[indices[:, None], :, permutations, :], theta_batch[indices]


def collate_fn_vec(batch, config, shuffling_method=0, debug=False):
    """
    batch: [
            (seqC, theta, probR),
            (seqC, theta, probR),
            ...
            (seqC, theta, probR),
            ]
            seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)

            shuffling_method: 0: complex shuffle - expand x from (B, D*M*S, 16) to (B*C, D*M*S, 16) then shuffle along the 2nd axis, then shuffle the batch BC
                              1: simple shuffle - shuffle x (B, D*M*S, 16) along the 2nd axis, then expand x to (B*C, D*M*S, 16)
    """

    C = config["dataset"]["num_probR_sample"]
    B = len(batch)

    if debug:
        start_time_0 = time.time()

    # Preallocate tensors
    seqC_batch = torch.empty((B, *batch[0][0].shape))
    theta_batch = torch.empty((B, *batch[0][1].shape))
    probR_batch = torch.empty((B, *batch[0][2].shape))

    # Fill tensors with data from the batch
    for i, (seqC, theta, probR) in enumerate(batch):
        seqC_batch[i] = seqC
        theta_batch[i] = theta
        probR_batch[i] = probR

    # seqC, theta, probR = zip(*batch)

    # seqC_batch = torch.stack(seqC)
    # theta_batch = torch.stack(theta)
    # probR_batch = torch.stack(probR)

    if debug:
        print(f"collate_fn_vec: dataloading {(time.time() - start_time_0)*1000:.2f} ms")

    del batch, seqC, theta, probR

    if shuffling_method == 0:
        # Repeat seqC and theta C times along a new dimension
        seqC_batch = seqC_batch.repeat_interleave(
            C, dim=0
        )  # (C*B, D*M*S, 15) first C samples are the same, from the first batch
        theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)

        # Repeat probR C times along a new dimension and sample from Bernoulli distribution
        probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
        # probR_batch = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)
        # probR_batch.bernoulli_() # (C*B, D*M*S, 1)

        # Concatenate x_seqC and x_choice
        x_batch = torch.cat(
            [seqC_batch, probR_batch.bernoulli_()], dim=-1
        )  # (C*B, D*M*S, 16)
        del probR_batch, seqC_batch
        gc.collect()

        if debug:
            print(
                f"\ncollate_fn_vec: get x_batch {(time.time() - start_time_0)*1000:.2f} ms"
            )
            start_time = time.time()

        # Shuffle x along the 2nd axis
        # x_batch = torch.stack([x_batch[i,:,:][torch.randperm(x_batch.shape[1]),:] for i in range(x_batch.shape[0])])
        DMS = x_batch.shape[1]
        # x_batch_shuffled = torch.empty_like(x_batch)

        # permutations = generate_permutations(B*C, DMS)
        # permutations = list(map(lambda _: torch.randperm(DMS), range(B*C)))

        # permutations = [torch.randperm(DMS) for _ in range(B*C)]
        permutations = torch.stack([torch.randperm(DMS) for _ in range(B * C)])
        # permutations = torch.rand(B*C, DMS).argsort(dim=-1)

        # if debug:
        #     print(f"\ncollate_fn_vec: generate permutations {(time.time() - start_time)*1000:.2f} ms")
        #     start_time = time.time()
        # start_time = time.time()

        # for i in range(B*C):
        # x_batch_shuffled[i] = x_batch[i][permutations[i]]
        # x_batch_shuffled[i] = x_batch[i][torch.randperm(DMS)]

        # gathering method
        # indices = torch.argsort(torch.rand(*x_batch.shape[:2]), dim=1)
        # x_batch_shuffled = torch.gather(x_batch, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, x_batch.shape[-1]))

        # del x_batch

        if debug:
            print(
                f"collate_fn_vec: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms"
            )
            start_time = time.time()

            # permutations = torch.stack([torch.randperm(DMS) for _ in range(B*C)])
            # x_batch_shuffled = x_batch[torch.arange(B * C)[:, None], permutations]
            # x_batch_shuffled_2 = x_batch[torch.arange(B * C)[:, None], permutations]
            # print(f"collate_fn_vec: shuffle x_batch_2 {(time.time() - start_time)*1000:.2f} ms")
            # # print(f'same? {torch.all(torch.eq(x_batch_shuffled, x_batch_shuffled_2))}')
            # start_time = time.time()

        # Shuffle the batched dataset
        # indices             = torch.randperm(x_batch_shuffled.shape[0])
        indices = torch.randperm(x_batch.shape[0])
        # x_batch_shuffled    = x_batch_shuffled[indices]
        # theta_batch         = theta_batch[indices]

        # if debug:
        #     print(f"collate_fn_vec: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        #     print(f"collate_fn_vec: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms")

        # return x_batch_shuffled[indices], theta_batch[indices]
        # shuffle along the 1st axis individually and then shuffle the batch
        # return x_batch[torch.arange(B * C)[:, None], permutations][indices], theta_batch[indices]
        return x_batch[indices[:, None], permutations], theta_batch[indices]

    elif shuffling_method == 1:
        # shuffle seqC_batch and theta_batch along the 2nd axis
        DMS = seqC_batch.shape[1]
        for i in range(B):
            indices = torch.randperm(DMS)
            seqC_batch[i] = seqC_batch[i][indices]
            probR_batch[i] = probR_batch[i][indices]

        theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)
        seqC_batch = seqC_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 15)
        probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
        probR_batch = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)

        x_batch = torch.cat([seqC_batch, probR_batch], dim=-1)  # (C*B, D*M*S, 16)
        del seqC_batch, probR_batch

        return x_batch, theta_batch


def collate_fn(batch, config, debug=False):
    C = config["dataset"]["num_probR_sample"]

    if debug:
        start_time_0 = time.time()

    x_batch, theta_batch = [], []

    x_batch = torch.empty(
        (
            C * len(batch),
            batch[0][0].shape[0],
            batch[0][0].shape[1] + batch[0][2].shape[1],
        )
    )
    theta_batch = torch.empty((C * len(batch), batch[0][1].shape[0]))

    for i, (seqC, theta, probR) in enumerate(
        batch
    ):  # seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
        probR = probR.unsqueeze_(dim=0).repeat_interleave(C, dim=0)  # (C, D*M*S, 1)
        x_seqC = seqC.unsqueeze_(dim=0).repeat_interleave(C, dim=0)  # (C, D*M*S, 15)
        x_choice = torch.bernoulli(probR)  # (C, D*M*S, 1)

        x = torch.cat([x_seqC, x_choice], dim=-1)
        theta = theta.unsqueeze_(dim=0).repeat_interleave(C, dim=0)  # (C, 4)

        x_batch[i * C : (i + 1) * C] = x
        theta_batch[i * C : (i + 1) * C] = theta

    if debug:
        print(f"\ncollate_fn: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")

    if debug:
        start_time = time.time()
    # Shuffle x along the 2nd axis
    x_batch = torch.stack(
        [x_batch[i][torch.randperm(x_batch.shape[1])] for i in range(x_batch.shape[0])]
    )
    if debug:
        print(f"collate_fn: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")

    if debug:
        start_time = time.time()
    # Shuffle the batched dataset
    indices = torch.randperm(x_batch.shape[0])
    x_batch = x_batch[indices]
    theta_batch = theta_batch[indices]
    if debug:
        print(f"collate_fn: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        print(
            f"collate_fn: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms"
        )

    return x_batch.to(torch.float32), theta_batch.to(torch.float32)


def collate_fn_probR(batch, Rchoice_method="probR_sampling", num_probR_sample=10):
    """OLD VERSION
    batch is a list of tuples, each tuple is (theta, x, prior_masks)
        original shapes:
        theta.shape = (T*C, L_theta)
        x.shape     = (T*C, DMS, L_x)
        (sequence should be shuffled when batched into the dataloader)

        e.g. C = 1  if Rchoice_method == 'probR'
        e.g. C = 10 if Rchoice_method == 'probR_sampling'
    """

    theta, x, _ = zip(*batch)

    theta = torch.stack(theta)
    x = torch.stack(x)
    _ = torch.stack(_)

    if Rchoice_method == "probR":
        # repeat theta and x for each probR sample
        theta_new = theta.repeat_interleave(num_probR_sample, dim=0)  # (T*C, L_theta)
        x_new = x.repeat_interleave(num_probR_sample, dim=0)  # (T*C, DMS, 15+1)
        _ = _.repeat_interleave(num_probR_sample, dim=0)  # (T*C, 1)
        x_seqC = x_new[:, :, :-1]  # (T*C, DMS, 15)
        x_probRs = x_new[:, :, -1].unsqueeze_(dim=2)  # (T*C, DMS, 1)

        # sample Rchoice from probR with Bernoulli
        x_Rchoice = torch.bernoulli(x_probRs)
        x = torch.cat((x_seqC, x_Rchoice), dim=2)
        theta = theta_new

    return theta, x, _
