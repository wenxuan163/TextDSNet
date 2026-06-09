import torch


def collate_fn(batch):
    images, texts, targets = zip(*batch)  # 将batch中的数据解包成images和targets两个元组

    # 将images堆叠成一个批次
    images = torch.stack(images)

    # 初始化一个新的targets列表，每个元素都是一个字典
    new_targets = []
    for target in targets:
        new_target = {
            # 'name': target['name'],
            'labels': target['labels'],
            'boxes': target['boxes'],
            'masks': target['masks']
        }
        new_targets.append(new_target)

    return images, texts, new_targets
