import os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from options.train_options import TrainOptions
from dataset import create_dataset
from model import create_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:128"

if __name__ == '__main__':
    opt = TrainOptions().parse()  # 获取训练选项
    dataset = create_dataset(opt)  # 根据opt.dataset_mode和其他选项创建数据集
    model = create_model(opt)  # 根据opt.model和其他选项创建模型
    model.setup(opt)  # 加载和打印网络；创建调度器

    # 加载预训练模型
    if opt.load_trained_model:
        model.load_networks('latest_net_G.pth')  # 假设的方法，根据你的模型类进行调整

    model.train()
    print(model)

    for epoch in range(0, opt.niter + opt.epoch_count + 1):
        model.update_learning_rate()  # 更新学习率
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            model.set_input(data)  # 从数据加载器中解压数据
            model.optimize_parameters()  # 进行一次前向传播和反向传播

            if i % 1 == 0:  # 打印训练信息
                errors = model.get_current_losses()
                loss_value = errors['G_A'] + errors['G_B'] + errors['D_A'] + errors['D_B']
                print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' %
                      (epoch, opt.niter + opt.epoch_count, i, len(dataset), loss_value))

        if epoch % opt.save_epoch_freq == 0:  # 保存模型权重
            print('Saving the model at the end of epoch %d' % epoch)
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.epoch_count, time.time() - epoch_start_time))

    model.save_networks('latest')  # 保存最新的模型权重
    model.save_networks(epoch)  # 保存最后一个epoch的模型权重
