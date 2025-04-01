import torch

# history = 8
# obs_nums = 10
# self_obs = torch.zeros(4,history*obs_nums) 
# print(self_obs.shape)

# obs = torch.rand(4,obs_nums)
# print(obs)
# print(self_obs[:,obs_nums:].shape)
# self_obs = torch.cat([self_obs[:,obs_nums:],obs],dim=1)
# print(self_obs)

# def replace_if_all_zero(big_tensor, small_tensor):
#     if torch.all(big_tensor == 0):
#         return small_tensor[:, None, :].expand_as(big_tensor)
#     return big_tensor

# num_envs = 2
# content_len = 3
# num_obs = 4

# # 创建一个全0的大张量
# big_tensor = torch.zeros(num_envs, content_len, num_obs)
# # 创建一个小张量
# small_tensor = torch.randn(num_envs, num_obs)
# print(small_tensor)

# if torch.all(big_tensor == 0):
#     result = small_tensor[:, None, :].expand_as(big_tensor)

# # result = replace_if_all_zero(big_tensor, small_tensor)
# print(result)  # 应该输出 torch.Size([2, 3, 4])



list = [1,2,3,4]
print(list[-3:-1])