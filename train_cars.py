import functools
import os
import numpy as np
import scipy.io as sio
import train_app
from datasets import cars
from datasets import util
import nets.deep_sort.network_definition as net


class Car:
    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        images, ids = cars.read_train_split_to_image(self._dataset_dir)
        images = np.array(images)
        ids = np.array(ids)
        train_indices, _ = util.create_validation_split(np.asarray(ids, np.int64),
                            self._num_validation_y, self._seed)
        images = images[train_indices]
        ids = ids[train_indices]
        return images, ids

    def read_validation(self):
        images, ids = cars.read_train_split_to_image(self._dataset_dir)
        _, valid_indices = util.create_validation_split(np.asarray(ids, np.int64),
                            self._num_validation_y, self._seed)

        images = images[valid_indices]
        ids = ids[valid_indices]
        return images, ids

def main():
    arg_parser = train_app.create_default_argument_parser("cars")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to Cars dataset directory.",
        default="clips/pictures")
    args = arg_parser.parse_args()
    dataset = Car(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y = dataset.read_train()
        print("Train set size: %d images, %d identites" % (
            len(train_x), len(np.unique(train_y))))

        network_factory = net.create_network_factory(
            is_training=True, num_classes=cars.MAX_LABEL +1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=cars.IMAGE_SHAPE,
            **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y = dataset.read_validation()
        print("Validation set size: %d images, %d identites" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=cars.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, None,
            image_shape=cars.IMAGE_SHAPE, **eval_kwargs)
    elif args.mode == "export":
        raise NotImplementedError()
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=cars.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=cars.IMAGE_SHAPE,
            output_filename="./cars.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=cars.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=cars.IMAGE_SHAPE,
            output_filename="./cars.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    main()