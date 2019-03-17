import  sympy
import myfunc
import numpy as np
from imp import reload
import myfunc.formula as formula
from myfunc import formula as d
reload(myfunc)

d('2*x**x + sympy.exp(x)')

d(sympy.sin(sympy.pi/2))
list(range(10,1,-1))

x=sympy.Symbol('x')
f=2*x**x + np.exp(x)
sympy.latex(f)


sympy.sin(sympy.pi/2)



6//2*2==6

1/2

sympy.log(sympy.E)
a=sympy.sqrt(4)
a=2
sympy.root(8,3)

sympy.factorial(4)

x = sympy.Symbol('x')
fx = 2*x + 1

type(fx)


# 用evalf函数，传入变量的值，对表达式进行求值
fx.evalf(subs={x:2})

x,y = sympy.symbols('x y')

f = 2 * x + y

f.evalf(subs = {x:1,y:2})

f.evalf(subs = {x:1})
2.0*x + y

x = sympy.Symbol('x')

sympy.solve(x - 1,x)

sympy.solve(x ** 2 - 1,x)

sympy.solve(x ** 2 + 1,x)

x,y = sympy.symbols('x y')

##########代数方程
sympy.solve([x + y - 1,x - y -3],[x,y])
{x: 2, y: -1}





n = sympy.Symbol('n')

sympy.summation(2 * n,(n,1,100))\





x = sympy.Symbol('x')

f1 = sympy.sin(x)/x
myfunc.formula(f1)

sympy.limit(f1,x,0)


f2 = (1+x)**(1/x)

sympy.limit(f2,x,0)
E

f3 = (1+1/x)**x

sympy.limit(f3,x,sympy.oo)
E

# 求导,微分
x = sympy.Symbol('x')

f = x ** 2 + 2 * x + 1

sympy.diff(f,x)
f.diff(x)
f.diff(x,x)
2*x + 2

f2 = sympy.sin(x)

sympy.diff(f2,x)



y = sympy.Symbol('y')
del x
f3 = x**2 + 2*x + y**3
d('x**2 + 2*x + y**3')
# exec('a=x**2 + 2*x + y**3')
#
# f='x**2 + 2*x + y**3'
# e=eval(f)
# t=f
# try:
#     eval(f)
# except Exception as e:
#     a=str(e).split("'")[1]
#
#     exec(a+'=sympy.Symbol(a)')

d(x**2 + 2*x + y**3)
sympy.latex(f3)

sympy.diff(f3,x)
2*x + 2

sympy.diff(f3,y)
3*y**2
#############微分方程
x=sympy.Symbol('x')
f=sympy.Function('f')
sympy.dsolve(f(x).diff(x, x) + f(x), f(x))
d(sympy.dsolve(f(x).diff(x, x) + f(x), f(x)))


################# 积分
f=2*x
sympy.integrate(f,(x,0,1))

t,x = sympy.symbols('t x')

t,x = sympy.symbols('t x')
f = 2 * t

g = sympy.integrate(f,(t,0,x))

sympy.integrate(g,(x,0,3))
9


x = sympy.Symbol('x')

f = sympy.E ** x + 2 * x

sympy.integrate(f,x)

x**2 + exp(x)

f=1/(sympy.sqrt(x-1)*(1+sympy.root(x-1,3)))
sympy.integrate(f,x)

f=1/(x*sympy.ln(x))

sympy.log(sympy.E,2)


a,b,x,y=sympy.symbols('a b x y')
f=(a**x-b**x)/x
sympy.limit(f,x,0)

f=(x**x-x)/(sympy.ln(x)-x+1)
sympy.limit(f,x,1)

f=(x-sympy.sin(x))/sympy.tan(x**3)

1/sympy.tan(1)


a = sympy.cos(x)**2 - sympy.sin(x)**2
b = sympy.cos(2*x)
a.equals(b)

(1 + x * y).subs([(x, sympy.pi), (y, 2)])

##########级数展开
sympy.cos(x).series(x, 0, 5)

#分解
sympy.sin(x+y).expand(trig=True)


# 矩阵

A=sympy.Matrix([[1,x],[y,1]])
A**2

import pprint
pprint.pprint(1/x)
sympy.pprint(1/x)

sympy.latex(1/x)

import pyglet
game_window = pyglet.window.Window(800, 600)

import latex
from sympy import Integral, preview

from sympy.abc import x

sympy.latex(x**2)
sympy.latex(Integral(x**2, x))
preview(Integral(x**2, x)) #doctest:+SKIP


from sympy import *
init_printing()
x, y = symbols("x,y")
sqrt(x**2+y**2)

from IPython.display import Latex
Latex(r"$\sqrt{x^2+y^2}$")
import IPython

from IPython.external.mathjax import install_mathjax

install_mathjax()

import sympy
x = sympy.Symbol('x')
f = sympy.E ** x + 2 * x
t=sympy.integrate(f,x)

t=sympy.latex(t)
sympy.latex(x**2)

t=t.center(len(t)+2,'$')

# list_i = list(t)    # str -> list
# list_i.insert(0, '$')   # 注意不用重新赋值
# list_i.insert(len(t)+1, '$')
# str_i = ''.join(list_i)    # list -> str



import matplotlib.pyplot as plt
fig = plt.figure()
ax=fig.add_subplot(111)
ax.set_xlim([1,7])
ax.set_ylim([1,5])
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.text(3,3,t,fontsize=28)
plt.show()


t="$\sqrt{x^2+y^2}$"