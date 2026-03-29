from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset


def main():
    dataset = UnifiedDataset(
        desired_data=["eupeds_eth-train"],
        centric="agent",
        desired_dt=0.4,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.PEDESTRIAN],
        num_workers=0,
        verbose=True,
        data_dirs={
            "eupeds_eth": "~/datasets/eth_ucy",
        },
    )

    print(f"\n# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    batch: AgentBatch
    for i, batch in enumerate(tqdm(dataloader, desc="Loading batches")):
        print(f"\nBatch {i}: agent_hist shape={batch.agent_hist.shape}, future shape={batch.agent_fut.shape}")
        if i >= 2:
            print("... (showing first 3 batches only)")
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
