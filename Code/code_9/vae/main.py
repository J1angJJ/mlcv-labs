import os
import argparse
import time
from train import train_model, load_data
from visualize import visualize_model


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="自编码器实验程序")
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'visualize', 'all'],
                        help='运行模式：训练、可视化或全部')
    parser.add_argument('--model', type=str, default='all',
                        choices=['basic', 'undercomplete', 'regularized', 'denoising', 'vae', 'conv', 'all'],
                        help='模型类型')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='数据集选择')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    args = parser.parse_args()

    # 确保目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 获取数据集信息
    _, _, input_shape = load_data(dataset=args.dataset, batch_size=1)

    # 确定要处理的模型列表
    if args.model == 'all':
        model_configs = [
            {"type": "basic", "name": "basic", "encoding_dim": 128, "dataset": args.dataset},
            {"type": "basic", "name": "undercomplete", "encoding_dim": 32, "dataset": args.dataset},
            {"type": "regularized", "name": "regularized", "encoding_dim": 128, "lambda_l1": 1e-4,
             "dataset": args.dataset},
            {"type": "denoising", "name": "denoising", "encoding_dim": 128, "noise_factor": 0.3,
             "dataset": args.dataset},
            {"type": "vae", "name": "vae", "encoding_dim": 20, "dataset": args.dataset}
        ]

        # 为CIFAR-10添加卷积自编码器
        if args.dataset == 'cifar10':
            model_configs.append({"type": "conv", "name": "conv", "encoding_dim": 128, "dataset": args.dataset})
    else:
        # 单个模型配置
        encoding_dim = 32 if args.model == "undercomplete" else 20 if args.model == "vae" else 128
        model_config = {
            "type": args.model,
            "name": args.model,
            "encoding_dim": encoding_dim,
            "dataset": args.dataset
        }
        if args.model == "regularized":
            model_config["lambda_l1"] = 1e-4
        if args.model == "denoising":
            model_config["noise_factor"] = 0.3

        model_configs = [model_config]

    # 根据模式执行
    if args.mode == 'train' or args.mode == 'all':
        total_start_time = time.time()
        print(f"\n===== 开始在{args.dataset}数据集上训练模型 =====")

        for config in model_configs:
            if config["type"] == "regularized":
                train_model(config["type"], config["dataset"], encoding_dim=config["encoding_dim"],
                            epochs=args.epochs, lambda_l1=config["lambda_l1"])
            elif config["type"] == "denoising":
                train_model(config["type"], config["dataset"], encoding_dim=config["encoding_dim"],
                            epochs=args.epochs, noise_factor=config["noise_factor"])
            else:
                train_model(config["type"], config["dataset"], encoding_dim=config["encoding_dim"],
                            epochs=args.epochs)

        print(f"\n所有训练完成！总耗时: {(time.time() - total_start_time) / 60:.2f} 分钟")

    if args.mode == 'visualize' or args.mode == 'all':
        print(f"\n===== 开始可视化{args.dataset}数据集上的模型 =====")

        for config in model_configs:
            visualize_model(config["type"], config["dataset"])

        print("\n所有可视化完成！")


if __name__ == "__main__":
    main()