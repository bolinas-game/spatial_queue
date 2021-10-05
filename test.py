import attr, cattr, time
a=open("test.txt","w")

def f1():
    a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in a_list:
        a.write(str(i)+"\n")
        time.sleep(5)
f1()
# def get():
#     global a_list
#     if len(a_list)>0:
#         for i in a_list:
#             print(i)
#             yield i
#
# if __name__=="__main__":
#     for j in get():
#         j
        # print(j)
    # b=get()
    # c=get()