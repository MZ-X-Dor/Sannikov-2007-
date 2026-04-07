# Replication of Sannikov (2008) Figure 1

This repository provides a numerical replication of **Figure 1** from:

> Sannikov, Y. (2008). *A Continuous-Time Version of the Principal-Agent Problem.*

---

## 📌 Overview

This project solves the continuous-time principal-agent model using the **HJB equation approach**. The key idea is to reduce the dynamic contracting problem to a **second-order ODE with boundary conditions**, where the state variable is the agent’s continuation value ( W ).

We numerically compute:

* The principal’s value function ( F(W) )
* Optimal effort ( a(W) )
* Optimal consumption ( c(W) )
* Drift of continuation value

---

## ⚙️ Model Setup

We follow the benchmark example in the paper:

* Utility:
  ( u(c) = \sqrt{c} )

* Cost of effort:
  ( h(a) = \frac{1}{2}a^2 + 0.4a )

* Parameters:
  ( r = 0.1, \quad \sigma = 1 )

* Retirement value:
  ( F_0(W) = -W^2 )

---

## 🧠 Methodology

### 1. HJB Equation

The value function ( F(W) ) satisfies:

[
F''(W) = \min_{a,c} \frac{F(W) - a + c - F'(W)(W - u(c) + h(a))}{r\gamma(a)^2\sigma^2/2}
]

with boundary conditions:

* ( F(0) = 0 )
* ( F(W_{gp}) = F_0(W_{gp}) )
* ( F'(W_{gp}) = F_0'(W_{gp}) ) (smooth pasting)

---

### 2. Shooting Method

We solve the boundary value problem using a **shooting method**:

1. Guess initial slope ( p_0 = F'(0) )
2. Solve the ODE forward
3. Detect whether the solution:

   * hits ( F_0 ) (valid path)
   * overshoots (never hits ( F_0 ))

We search for the **largest ( p_0 )** such that the solution still hits ( F_0 ).

This is implemented via:

* coarse grid search → find bracket
* bisection → refine ( p_0^* )

---

## 🔍 Numerical Results

### Shooting bracket

```
=== found bracket: (2.445810055865922, 2.4608938547486034) ===
```

### Optimal initial slope

```
Optimal p0 = F'(0) ≈ 2.4585505303
```

### Boundary conditions

```
Wgp ≈ 0.9607296198

Value matching:   F(Wgp) - F0(Wgp)  = 0.000000e+00
Smooth pasting:   p(Wgp) - F0'(Wgp) = -3.133048e-04
```

---

### Economic thresholds

```
W*   ≈ 0.1273664627   (maximum of F)
W**  ≈ 0.1287329254   (consumption starts)
```

---

### Policy statistics

```
max effort        = 0.793368
min effort        = 0.092516

max consumption   = 0.923302

min drift         = 0.000385
max drift         = 0.063206
```

---

## 📈 Interpretation

* The value function ( F(W) ) is **concave** and lies above ( F_0(W) )

* Optimal contract exhibits:

  * **probation region**: ( c(W) = 0 )
  * **interior region**: increasing consumption
  * **retirement boundary** at ( W_{gp} )

* Effort is **non-monotonic**, reflecting:

  * trade-off between incentive provision and risk

* Drift is positive over most of the domain, pushing the agent toward retirement

---

## ✅ Validation

The numerical solution satisfies:

* Value matching at ( W_{gp} )
* Smooth pasting condition (approximate)
* ( F(W) \ge F_0(W) ) along the path

These confirm consistency with **Theorem 1** in the paper.

---

## 🚀 Possible Extensions

* Improve resolution of ( p_0 ) (tighter smooth pasting)
* Alternative utility / cost functions
* Add outside options or promotion (Section 4 of the paper)
* Formal verification (e.g., Lean)

---

## 📂 Structure

```
Sannikov Figure 1.py   # main script
```

---

## 📬 Notes

This implementation focuses on clarity and economic structure rather than extreme numerical optimization. The code can be extended for more robust or high-precision solutions.

---
