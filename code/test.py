#print('hello' + ' what are you doing?')
#print('let\'s go')
#print('''hello
#do you
#remember me
#?''')
#变量
'''
greet = '你好，'
print(greet + '张三')
print(greet + '李四')
'''
# greet = '你好'
# greet_chinese = greet
# greet_english = 'hello'
# greet = greet_english
# print(greet)
# print(greet_chinese)
#字符型str
# s = 'hello world'
# print(len(s))
# print(s[len(s)-1])
# #
# #定义bool类型
# a = True #要大写
# b = False
# #空值类型
# c = None
# #type17
# print(type(s))
# print(type(a))
# print(type(c))
# print(type(1))
# print(type(1.1))
# print(type(1.22))
#input
# user_height = float(input('请输入你的身高（单位：米）: ')) #input 返回的都是字符串型
# user_weight = float(input('请输入你的体重（单位：千克）: '))
# BMI = user_weight/(user_height**2) #字符串不能运算
# print(BMI)
#-----if-----
# mood_index = int(input("请输入心情指数："))
# if mood_index >= 80:  #----要注意有个冒号，以及后面的缩进
#     print('心情不错呀')
# else:#---------------------和if是同级的缩进
#     print('开心点呀')
# user_height = float(input('请输入你的身高（单位：米）: ')) #input 返回的都是字符串型
# user_weight = float(input('请输入你的体重（单位：千克）: '))
# BMI = user_weight/(user_height**2) #字符串不能运算
# # print(BMI)
# if BMI < 18.5:
#     print('偏廋')
# elif BMI >= 18.5 and BMI <= 25:#----- 注意缩进
#     print('正常')
# elif BMI >= 25 and BMI <= 30:
#     print('偏胖')
# else:
#     print('太胖了')
#     print(BMI)
#----------------多条件判断-----------------
# mood_index = int(input('请输入你的心情指数：'))
# money = int(input('请输入你还有多少钱：'))
# friend = int(input('请输入你有几个朋友：'))
# # if mood_index >=80:
# #     if money >= 100:
# #         if friend >= 1:
# #             print('下馆子')
# #         else:
# #             print('去食堂')
# #     else:
# #         print('去食堂')
# # else:
# #     print('去食堂')
# if(mood_index >=80 and money >=100 and friend >=1):
#     print('下馆子')
# else:
#     print('去食堂')
# class Student:
#     def __init__(self,name,student_id):
#         self.name = name
#         self.student_id = student_id
#         self.grade = {"语文": 0,"数学": 0,"英语": 0}
#     def set_grade(self,course,grade):
#         if course in self.grade:
#             self.grade[course] = grade
#     def print_grade(self):
#         print(f"学生 {self.name} (学号 {self.student_id}) 成绩为：")
#         print(self.grade)
# zhang = Student("小张",852403109)
# zhang.set_grade("语文",100)
# zhang.set_grade("数学",100)
# zhang.set_grade("英语",99)
# zhang.print_grade()
class Employee:
    def __init__(self,name,id):
        self.name = name
        self.id = id
    def print_info(self):
        print(f"员工姓名:{self.name},工号: {self.id}")
class fulltimeEmployee(Employee):
    def __init__(self,name,id,salary):
        super().__init__(name,id)
        self.salary = salary
    def calculate_monthly_salary(self):
        return self.salary
class parttimeEmployee(Employee):
    def __init__(self,name,id,dailysalary,day):
        super().__init__(name,id)
        self.dailysalary = dailysalary
        self.day = day
    def calculate_monthly_salary(self):
        return self.dailysalary*self.day
# zhangsan = fulltimeEmployee("张三",01,8000)
# lisi = parttimeEmployee("李四",02,100,30)
zhangsan = fulltimeEmployee("张三",1,8000)    #整数不能以0开头，除非是8进制
lisi = parttimeEmployee("李四",2,100,30)
zhangsan.print_info()
lisi.print_info()
print(zhangsan.calculate_monthly_salary())
print(lisi.calculate_monthly_salary())






