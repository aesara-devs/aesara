
import aesara
a = aesara.tensor.vector("a") # declare variable
b = a + a**10                 # build symbolic expression
f = aesara.function([a], b)   # compile function
print(f([0,1,2]))
# prints `array([0,2,1026])`

aesara.printing.pydotprint(b, outfile="pics/f_unoptimized.png", var_with_name_simple=True)
aesara.printing.pydotprint(f, outfile="pics/f_optimized.png", var_with_name_simple=True)
