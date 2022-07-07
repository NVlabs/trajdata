from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.data_structures.map import Map


def plot_agent_batch(
    batch: AgentBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    agent_name: str = batch.agent_name[batch_idx]
    agent_type: AgentType = AgentType(batch.agent_type[batch_idx].item())
    ax.set_title(f"{str(agent_type)}/{agent_name}")

    history_xy: Tensor = batch.agent_hist[batch_idx].cpu()
    center_xy: Tensor = batch.agent_hist[batch_idx, -1, :2].cpu()
    future_xy: Tensor = batch.agent_fut[batch_idx, :, :2].cpu()

    if batch.maps is not None:
        agent_from_world_tf: Tensor = batch.agents_from_world_tf[batch_idx].cpu()
        world_from_raster_tf: Tensor = torch.linalg.inv(
            batch.rasters_from_world_tf[batch_idx].cpu()
        )

        agent_from_raster_tf: Tensor = agent_from_world_tf @ world_from_raster_tf

        patch_size: int = batch.maps[batch_idx].shape[-1]

        left_extent: float = (agent_from_raster_tf @ torch.tensor([0.0, 0.0, 1.0]))[
            0
        ].item()
        right_extent: float = (
            agent_from_raster_tf @ torch.tensor([patch_size, 0.0, 1.0])
        )[0].item()
        bottom_extent: float = (
            agent_from_raster_tf @ torch.tensor([0.0, patch_size, 1.0])
        )[1].item()
        top_extent: float = (agent_from_raster_tf @ torch.tensor([0.0, 0.0, 1.0]))[
            1
        ].item()

        ax.imshow(
            Map.to_img(
                batch.maps[batch_idx].cpu(),
                # [[0], [1], [2]]
                # [[0, 1, 2], [3, 4], [5, 6]],
            ),
            extent=(
                left_extent,
                right_extent,
                bottom_extent,
                top_extent,
            ),
            alpha=0.3,
        )

    ax.plot(
        history_xy[..., 0],
        history_xy[..., 1],
        c="orange",
        ls="--",
        label="Agent History",
    )
    ax.quiver(
        history_xy[..., 0],
        history_xy[..., 1],
        history_xy[..., -1],
        history_xy[..., -2],
        color="k",
    )
    ax.plot(future_xy[..., 0], future_xy[..., 1], c="violet", label="Agent Future")
    ax.scatter(center_xy[0], center_xy[1], s=20, c="orangered", label="Agent Current")

    num_neigh = batch.num_neigh[batch_idx]
    if num_neigh > 0:
        neighbor_hist = batch.neigh_hist[batch_idx]
        neighbor_fut = batch.neigh_fut[batch_idx]

        ax.plot([], [], c="olive", ls="--", label="Neighbor History")
        for n in range(num_neigh):
            ax.plot(neighbor_hist[n, :, 0], neighbor_hist[n, :, 1], c="olive", ls="--")

        ax.plot([], [], c="darkgreen", label="Neighbor Future")
        for n in range(num_neigh):
            ax.plot(neighbor_fut[n, :, 0], neighbor_fut[n, :, 1], c="darkgreen")

        ax.scatter(
            neighbor_hist[:num_neigh, -1, 0],
            neighbor_hist[:num_neigh, -1, 1],
            s=20,
            c="gold",
            label="Neighbor Current",
        )

    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        ax.plot(
            batch.robot_fut[batch_idx, 1:, 0],
            batch.robot_fut[batch_idx, 1:, 1],
            label="Ego Future",
            c="blue",
        )
        ax.scatter(
            batch.robot_fut[batch_idx, 0, 0],
            batch.robot_fut[batch_idx, 0, 1],
            s=20,
            c="blue",
            label="Ego Current",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    ax.axis("equal")

    if show:
        plt.show()

    if close:
        plt.close()


def plot_scene_batch(
    batch: SceneBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    num_agents: int = batch.num_agents[batch_idx].item()

    history_xy: Tensor = batch.agent_hist[batch_idx].cpu()
    center_xy: Tensor = batch.agent_hist[batch_idx, ..., -1, :2].cpu()
    future_xy: Tensor = batch.agent_fut[batch_idx, ..., :2].cpu()

    if batch.maps is not None:
        centered_agent_id: int = 0
        agent_from_world_tf: Tensor = batch.centered_agent_from_world_tf[
            batch_idx
        ].cpu()
        world_from_raster_tf: Tensor = torch.linalg.inv(
            batch.rasters_from_world_tf[batch_idx, centered_agent_id].cpu()
        )

        agent_from_raster_tf: Tensor = agent_from_world_tf @ world_from_raster_tf

        patch_size: int = batch.maps[batch_idx, centered_agent_id].shape[-1]

        left_extent: float = (agent_from_raster_tf @ torch.tensor([0.0, 0.0, 1.0]))[
            0
        ].item()
        right_extent: float = (
            agent_from_raster_tf @ torch.tensor([patch_size, 0.0, 1.0])
        )[0].item()
        bottom_extent: float = (
            agent_from_raster_tf @ torch.tensor([0.0, patch_size, 1.0])
        )[1].item()
        top_extent: float = (agent_from_raster_tf @ torch.tensor([0.0, 0.0, 1.0]))[
            1
        ].item()

        ax.imshow(
            Map.to_img(
                batch.maps[batch_idx, centered_agent_id].cpu(),
                # [[0], [1], [2]]
                # [[0, 1, 2], [3, 4], [5, 6]],
            ),
            extent=(
                left_extent,
                right_extent,
                bottom_extent,
                top_extent,
            ),
            alpha=0.3,
        )

    for agent_id in range(num_agents):
        ax.plot(
            history_xy[agent_id, ..., 0],
            history_xy[agent_id, ..., 1],
            c="orange",
            ls="--",
            label="Agent History" if agent_id == 0 else None,
        )
        ax.quiver(
            history_xy[agent_id, ..., 0],
            history_xy[agent_id, ..., 1],
            history_xy[agent_id, ..., -1],
            history_xy[agent_id, ..., -2],
            color="k",
        )
        ax.plot(
            future_xy[agent_id, ..., 0],
            future_xy[agent_id, ..., 1],
            c="violet",
            label="Agent Future" if agent_id == 0 else None,
        )
        ax.scatter(
            center_xy[agent_id, 0],
            center_xy[agent_id, 1],
            s=20,
            c="orangered",
            label="Agent Current" if agent_id == 0 else None,
        )

    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        ax.plot(
            batch.robot_fut[batch_idx, 1:, 0],
            batch.robot_fut[batch_idx, 1:, 1],
            label="Ego Future",
            c="blue",
        )
        ax.scatter(
            batch.robot_fut[batch_idx, 0, 0],
            batch.robot_fut[batch_idx, 0, 1],
            s=20,
            c="blue",
            label="Ego Current",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    ax.axis("equal")

    if show:
        plt.show()

    if close:
        plt.close()
