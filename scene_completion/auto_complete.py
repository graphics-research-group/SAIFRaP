import argparse
import logging
import os
import sys
import numpy as np
import torch
import trimesh

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, \
    poll_specific_class, make_network_input, render_to_folder, \
    render_scene_from_bbox_params

from scene_synthesis.datasets import get_dataset_raw_and_encoded, filter_function
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network

from simple_3dviz import Scene

def main(argv):
    parser = argparse.ArgumentParser(description="Complete a partially complete scene")

    parser.add_argument("config_file", help="Path to the file that contains the experiment configuration")
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument("path_to_pickled_3d_futute_models", help="Path to the 3D-FUTURE model meshes")
    parser.add_argument("path_to_floor_plan_textures", help="Path to floor texture images")
    parser.add_argument("--weight_file", default=None, help="Path to a pretrained model")
    parser.add_argument("--n_sequences", default=1, type=int, help="The number of sequences to be generated")
    parser.add_argument("--background", type=lambda x: list(map(float, x.split(","))), default="1,1,1,1", help="Set the background of the scene")
    parser.add_argument("--up_vector", type=lambda x: tuple(map(float, x.split(","))), default="0,1,0", help="Up vector of the scene")
    parser.add_argument("--camera_position", type=lambda x: tuple(map(float, x.split(","))), default="-0.10923499,1.9325259,-7.19009", help="Camera position in the scene")
    parser.add_argument("--camera_target", type=lambda x: tuple(map(float, x.split(","))), default="1,0,1", help="Set the target for the camera")
    parser.add_argument("--window_size", type=lambda x: tuple(map(int, x.split(","))), default="512,512", help="Define the size of the scene and the window")
    parser.add_argument("--with_rotating_camera", action="store_true", help="Use a camera rotating around the object")
    parser.add_argument("--save_frames", action="store_true", help="Path to save the visualization frames to")
    parser.add_argument("--n_frames", type=int, default=360, help="Number of frames to be rendered")
    parser.add_argument("--without_screen", action="store_true", help="Perform no screen rendering")
    parser.add_argument("--scene_id", default=None, help="The scene id to be used for conditioning")
    parser.add_argument("--fixed_class_labels", nargs="+", default=[], help="List of fixed object class names to be added")
    parser.add_argument("--fixed_translations", nargs="+", default=[], help="List of fixed translations as comma-separated 3D positions")

    args = parser.parse_args(argv)

    logging.getLogger("trimesh").setLevel(logging.ERROR)
    device = torch.device("cpu")

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(config["data"], split=config["training"].get("splits", ["train", "val"])),
        split=config["training"].get("splits", ["train", "val"])
    )

    objects_dataset = ThreedFutureDataset.from_pickled_dataset(args.path_to_pickled_3d_futute_models)
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(config["data"], split=config["validation"].get("splits", ["test"])),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types".format(len(dataset), dataset.n_object_types))

    object_types = np.array(dataset.object_types)
    fixed_insertions = []
    if args.fixed_class_labels and args.fixed_translations:
        if len(args.fixed_class_labels) != len(args.fixed_translations):
            raise ValueError("Mismatch between number of class labels and translations")
        for label, t_str in zip(args.fixed_class_labels, args.fixed_translations):
            if label not in object_types:
                raise ValueError(f"Unknown class label: {label}")
            index = np.where(object_types == label)[0][0]
            t = np.array(list(map(float, t_str.split(","))))
            if t.shape[0] != 3:
                raise ValueError("Each translation must be a 3D vector")
            fixed_insertions.append((index, t))
    print("You are keeping the following fixed insertions: " +
          str([f"{object_types[idx]} @ {tuple(pos)}" for idx, pos in fixed_insertions]))

    network, _, _ = build_network(dataset.feature_size, dataset.n_classes, config, args.weight_file, device=device)
    network.eval()

    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    classes = np.array(dataset.class_labels)
    for i in range(args.n_sequences):
        scene_idx = given_scene_id or np.random.choice(len(dataset))
        current_scene = raw_dataset[scene_idx]
        current_boxes = dataset[scene_idx]
        print("{} / {}: Using the {} floor plan of scene {}".format(i, args.n_sequences, scene_idx, current_scene.scene_id))

        floor_plan, tr_floor, room_mask = floor_plan_from_scene(current_scene, args.path_to_floor_plan_textures)

        # Start from an empty layout and add fixed objects manually
        boxes = {
            "class_labels": torch.empty((1, 0, dataset.n_classes), dtype=torch.float32),
            "translations": torch.empty((1, 0, 3), dtype=torch.float32),
            "sizes": torch.empty((1, 0, 3), dtype=torch.float32),
            "angles": torch.empty((1, 0, 1), dtype=torch.float32),
        }

        updated_boxes = boxes
        if fixed_insertions:
            print("Adding fixed objects before scene completion")
            for class_index, translation in fixed_insertions:
                inserted = network.add_object(
                    room_mask=room_mask,
                    class_label=class_index,
                    boxes=updated_boxes,
                    translation=translation
                )
                updated_boxes = {k: v[:, 1:-1, :] for k, v in inserted.items()}

        render_to_folder(
            args,
            "partial_{}_{:03}".format(current_scene.scene_id, i),
            dataset,
            objects_dataset,
            tr_floor,
            floor_plan,
            scene,
            updated_boxes,
            True
        )

        print("Completing scene after inserting fixed objects (if any)")
        bbox_params = network.complete_scene(
            boxes=updated_boxes,
            room_mask=room_mask
        )

        path_to_image = f"{args.output_directory}/{current_scene.scene_id}_{scene_idx}_{i}"
        path_to_objs = os.path.join(
            args.output_directory,
            f"complete_{current_scene.scene_id}_{i:03d}"
        )
        render_scene_from_bbox_params(
            args,
            bbox_params,
            dataset,
            objects_dataset,
            classes,
            floor_plan,
            tr_floor,
            scene,
            path_to_image,
            path_to_objs
        )

if __name__ == "__main__":
    main(sys.argv[1:])