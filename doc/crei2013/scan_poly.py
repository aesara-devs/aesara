
import numpy as np

import aesara
import aesara.tensor as tt

coefficients = aesara.tensor.vector("coefficients")
x = tt.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = aesara.tensor.arange(max_coefficients_supported)
components, updates = aesara.scan(fn=lambda coeff, power, free_var:
                                  coeff * (free_var ** power),
                                  outputs_info=None,
                                  sequences=[coefficients, full_range],
                                  non_sequences=x)
polynomial = components.sum()
calculate_polynomial = aesara.function(inputs=[coefficients, x],
                                       outputs=polynomial)

test_coeff = np.asarray([1, 0, 2], dtype=np.float32)
print(calculate_polynomial(test_coeff, 3))
# 19.0
