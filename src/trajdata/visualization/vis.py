from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.maps import RasterizedMap


def draw_agent(
    ax: Axes,
    agent_type: AgentType,
    agent_state: Tensor,
    agent_extent: Tensor,
    agent_to_world_tf: Tensor,
    **kwargs,
) -> None:
    """Draws a path with the correct location, heading, and dimensions onto the given axes

    Args:
        ax (Axes): _description_
        agent_type (AgentType): _description_
        agent_state (Tensor): _description_
        agent_extent (Tensor): _description_
        agent_to_world_tf (Tensor): _description_
    """

    if torch.any(torch.isnan(agent_extent)):
        if agent_type == AgentType.VEHICLE:
            length = 4.3
            width = 1.8
        elif agent_type == AgentType.PEDESTRIAN:
            length = 0.5
            width = 0.5
        elif agent_type == AgentType.BICYCLE:
            length = 1.9
            width = 0.5
        else:
            length = 1.0
            width = 1.0
    else:
        length = agent_extent[0].item()
        width = agent_extent[1].item()

    patch = FancyBboxPatch(
        [-length / 2, -width / 2], length, width, boxstyle="rarrow", **kwargs
    )
    transform = (
        mtransforms.Affine2D()
        .rotate(np.arctan2(agent_state[-2].item(), agent_state[-1].item()))
        .translate(agent_state[0], agent_state[1])
        + mtransforms.Affine2D(matrix=agent_to_world_tf.cpu().numpy())
        + ax.transData
    )
    patch.set_transform(transform)

    center_patch = Circle([0, 0], radius=0.25, **kwargs)
    center_patch.set_transform(transform)

    ax.add_patch(patch)
    ax.add_patch(center_patch)


def draw_history(
    ax,
    agent_type,
    agent_history,
    agent_extent,
    agent_to_world_tf,
    start_alpha=0.2,
    end_alpha=0.5,
    **kwargs,
):
    T = agent_history.shape[0]
    alphas = np.linspace(start_alpha, end_alpha, T)
    for t in range(T):
        draw_agent(
            ax,
            agent_type,
            agent_history[t],
            agent_extent,
            agent_to_world_tf,
            alpha=alphas[t],
            **kwargs,
        )


def draw_map(ax: Axes, map: Tensor, base_frame_from_map_tf: Tensor, **kwargs):
    patch_size: int = map.shape[-1]
    map_array = RasterizedMap.to_img(map.cpu())
    brightened_map_array = map_array * 0.2 + 0.8

    im = ax.imshow(
        brightened_map_array, extent=[0, patch_size, patch_size, 0], **kwargs
    )
    transform = (
        mtransforms.Affine2D(matrix=base_frame_from_map_tf.cpu().numpy()) + ax.transData
    )
    im.set_transform(transform)


def plot_agent_batch_all(
    batch: AgentBatch,
    ax: Optional[Axes] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    # Use first agent as common reference frame
    base_frame_from_world_tf = batch.agents_from_world_tf[0].cpu()

    # plot maps over each other with proper transformations:
    for i in range(len(batch.agent_name)):
        base_frame_from_map_tf = base_frame_from_world_tf @ torch.linalg.inv(
            batch.rasters_from_world_tf[i].cpu()
        )
        draw_map(ax, batch.maps[i], base_frame_from_map_tf, alpha=1.0)

    for i in range(len(batch.agent_name)):
        agent_type = batch.agent_type[i]
        agent_name = batch.agent_name[i]
        agent_hist = batch.agent_hist[i, :, :].cpu()
        agent_fut = batch.agent_fut[i, :, :].cpu()
        agent_extent = batch.agent_hist_extent[i, -1, :].cpu()
        base_frame_from_agent_tf = base_frame_from_world_tf @ torch.linalg.inv(
            batch.agents_from_world_tf[i].cpu()
        )

        palette = sns.color_palette("husl", 4)
        if agent_type == AgentType.VEHICLE:
            color = palette[0]
        elif agent_type == AgentType.PEDESTRIAN:
            color = palette[1]
        elif agent_type == AgentType.BICYCLE:
            color = palette[2]
        else:
            color = palette[3]

        transform = (
            mtransforms.Affine2D(matrix=base_frame_from_agent_tf.numpy()) + ax.transData
        )
        draw_history(
            ax,
            agent_type,
            agent_hist[:-1, :],
            agent_extent,
            base_frame_from_agent_tf,
            color=color,
            linewidth=0,
        )
        ax.plot(
            agent_hist[:, 0],
            agent_hist[:, 1],
            linestyle="-",
            color=color,
            transform=transform,
        )
        draw_agent(
            ax,
            agent_type,
            agent_hist[-1, :],
            agent_extent,
            base_frame_from_agent_tf,
            facecolor=color,
            edgecolor="k",
        )
        ax.plot(
            agent_fut[:, 0],
            agent_fut[:, 1],
            linestyle="--",
            color=color,
            transform=transform,
        )

    ax.set_ylim(-30, 40)
    ax.set_xlim(-30, 40)
    ax.grid(False)

    if show:
        plt.show()

    if close:
        plt.close()


def plot_agent_batch(
    batch: AgentBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    legend: bool = True,
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
            RasterizedMap.to_img(
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
    # ax.quiver(
    #     history_xy[..., 0],
    #     history_xy[..., 1],
    #     history_xy[..., -1],
    #     history_xy[..., -2],
    #     color="k",
    # )

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
    ax.axis("equal")

    # Doing this because the imshow above makes the map origin at the top.
    ax.invert_yaxis()

    if legend:
        ax.legend(loc="best", frameon=True)

    if show:
        plt.show()

    if close:
        plt.close()

    return ax


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
            RasterizedMap.to_img(
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

    # Doing this because the imshow above makes the map origin at the top.
    ax.invert_yaxis()

    if show:
        plt.show()

    if close:
        plt.close()
