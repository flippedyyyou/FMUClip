import torch

def save_model(model, optimizer, epoch, eval_result, file_path):
    """
    保存模型和优化器的状态字典，以及评估结果
    :param model: 当前模型
    :param optimizer: 当前优化器
    :param epoch: 当前训练的 epoch
    :param eval_result: 当前评估指标（用于记录）
    :param file_path: 保存模型的路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eval_result': eval_result,
    }
    torch.save(checkpoint, file_path)
    print(f"Best model saved to {file_path}")

def load_model(model, optimizer, file_path):
    """
    加载模型和优化器的状态字典
    :param model: 初始化的模型
    :param optimizer: 初始化的优化器
    :param file_path: 模型保存路径
    :return: 加载后的模型和优化器
    """
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    eval_result = checkpoint['eval_result']
    print(f"Model loaded from {file_path}, starting from epoch {epoch}")
    return model, optimizer, epoch, eval_result
