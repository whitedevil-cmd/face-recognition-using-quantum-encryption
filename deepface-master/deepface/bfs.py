class Node:
  def __init__(self,val=0,next=None):
    self.val=val
    self.next=next
def split_arr(size,k):
  rem=size%k
  base=size//k
  return [base +1 if i<rem else base for i in range(k)]
def create_list(values):
  if not values:
    return None
  head=Node(values[0])
  current=head
  for i in values[1:]:
    current.next=Node(i)
    current=current.next
  return head
def print_list(indexes):
  for head in indexes:
    if head is None:
      print([])
      continue
    current=head
    vals=[]
    while current:
      vals.append(current.val)
      current=current.next
    print(f"{vals}",end=",")
def split(head,size,k):
  arr=split_arr(size,k)
  result=[]
  current=head
  for parts in arr:
    result.append(current)
    prev=None
    for _ in range(parts):
      if current:
        prev=current
        current=current.next
    if prev:
      prev.next=None
  return result
values=list(map(int,input().strip().split()))
k=int(input().strip())
head=create_list(values)
arr=split(head,len(values),k)
print_list(arr)

