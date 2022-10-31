


def swap(nums,left,right):
    temp = nums[left]
    nums[left] = nums[right]
    nums[right] = temp


nums = [2,0,2,1,1,0]
left = 0
right = len(nums) - 1
while (left <= right):
    if nums[left] > 1:
        swap(nums,left,right)
        right -= 1
    elif nums[left] <= 1:
        left += 1



print(nums)

