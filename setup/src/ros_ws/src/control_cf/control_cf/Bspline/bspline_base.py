### Naive approach for building of B-spline basis functions
import numpy as np
def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2

# def B(x, k, i, t):
#     x = np.asarray(x)   # x is an array
#     if k == 0:
#         # return np.where(1.0 if t[i] <= x < t[i+1] else 0.0)
#         return np.where((t[i] <= x) & (x< t[i+1]), 1.0, 0.0)
#     if t[i+k] == t[i]:
#         # c1 = 0.0
#         c1 = 0.0 + np.zeros_like(x)
#       #   print(c1)
#     else:
#         c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#         print("c1 = ",c1)
#     if t[i+k+1] == t[i+1]:
#         # c2 = 0.0
#         c2 = 0.0 + np.zeros_like(x)
#     else:
#         c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#         print("c2 = ",c2)
#     return c1 + c2

def bspline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))


if __name__=="__main__":
   k = 2
   t = [0, 1, 2, 3, 4, 5, 6]
   c = [-1, 2, 0, -1]

   x_eval = bspline(np.array([2.5,6]), t, c, k)
   bs = B(np.array([2.5,6]), k,1, t)

   print(x_eval)