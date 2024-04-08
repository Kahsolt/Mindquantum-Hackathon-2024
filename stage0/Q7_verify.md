> 评测机过不了，我感觉数据可能有问题，写公式验证一下...


### Q7 Solution Maths

According to the quantum Rn-Gate definition:

$$
\begin{array}{ll}
{\rm Rn}(\alpha, \beta, \gamma) 
&= e^{-i(\alpha \sigma_x + \beta \sigma_y + \gamma \sigma_z)/2} \\
&= {\rm cos}(f/2) I - i {\rm sin}(f/2) (\alpha \sigma_x + \beta \sigma_y + \gamma \sigma_z)/f
\end{array}
$$

where $ f = \sqrt{\alpha^2 + \beta^2+ \gamma^2} $, and $ I, \sigma_x, \sigma_y, \sigma_z $ are [pualis](https://lewisla.gitbook.io/learning-quantum/quantum-circuits/single-qubit-gates), Then the matrix form is expanded as:

$$
\begin{array}{ll}
&= {\rm cos}(\frac{f}{2})
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
\end{bmatrix} - \frac{i}{f}{\rm sin}(\frac{f}{2})
\begin{bmatrix}
\gamma & \alpha - \beta i \\
\alpha + \beta i & -\gamma \\
\end{bmatrix}
\end{array} \\
\begin{array}{ll}
&= \begin{bmatrix}
{\rm cos}(\frac{f}{2})-\frac{i}{f}{\rm sin}(\frac{f}{2})\gamma & -\frac{i}{f}{\rm sin}(\frac{f}{2})(\alpha - \beta i) \\
-\frac{i}{f}{\rm sin}(\frac{f}{2})(\alpha + \beta i) & {\rm cos}(\frac{f}{2})+\frac{i}{f}{\rm sin}(\frac{f}{2})\gamma \\
\end{bmatrix}
\end{array}
$$

And when it operates on $ \left| 0 \right> $, we just project on the first column:

$$
\begin{array}{ll}
\left| \psi \right>
= {\rm Rn}(\alpha, \beta, \gamma) \left| 0 \right>
= \begin{bmatrix}
{\rm cos}(\frac{f}{2})-\frac{i}{f}{\rm sin}(\frac{f}{2})\gamma \\
-\frac{i}{f}{\rm sin}(\frac{f}{2})(\alpha + \beta i) \\
\end{bmatrix}
= \begin{bmatrix}
u \\
v \\
\end{bmatrix}
\end{array}
$$

Now the original problem turns out to be calculating the expectation of $ E(\alpha, \beta, \gamma) $:

$$
\begin{array}{ll}
E(\alpha, \beta, \gamma) = \left< \psi | Z | \psi \right>
= \begin{bmatrix}
u^\dagger & v^\dagger
\end{bmatrix}
\begin{bmatrix}
1 &  0 \\
0 & -1 \\
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
\end{bmatrix}
= u^\dagger u - v^\dagger v
\end{array}
$$

$$
\begin{array}{ll}
&= \left[ {\rm cos}(\frac{f}{2})+\frac{i}{f}{\rm sin}(\frac{f}{2})\gamma \right] \left[ {\rm cos}(\frac{f}{2})-\frac{i}{f}{\rm sin}(\frac{f}{2})\gamma \right] - \left[ \frac{1}{f}{\rm sin}(\frac{f}{2})(\beta - \alpha i) \right] \left[ \frac{1}{f}{\rm sin}(\frac{f}{2})(\beta + \alpha i) \right] \\
&= {\rm cos}^2(\frac{f}{2}) + \frac{\gamma^2-\alpha^2-\beta^2}{f^2} {\rm sin}^2(\frac{f}{2})
\end{array}
$$

To get only the partial derivative of $ \frac{\partial E}{\partial \alpha} $ at point $ (1,2,3) $, we'd better fix $ \beta = 2 $ and $ \gamma = 3 $, reducing it to a partial function $ E_\alpha(\alpha) $:

$$
\begin{array}{ll}
E_\alpha(\alpha)
&= E(\alpha, 2, 3) \\
&= {\rm cos}^2(\frac{\sqrt{\alpha^2 + 4 + 9}}{2}) + \frac{9 - \alpha^2 - 4}{\alpha^2 + 4 + 9} {\rm sin}^2(\frac{\sqrt{\alpha^2 + 4 + 9}}{2}) \\
&= {\rm cos}^2(\frac{\sqrt{\alpha^2 + 13}}{2}) + \frac{5 - \alpha^2}{13 + \alpha^2} {\rm sin}^2(\frac{\sqrt{\alpha^2 + 13}}{2}) \\
&= 1 - \left( 2 - \frac{18}{\alpha^2 + 13} \right) {\rm sin}^2(\frac{\sqrt{\alpha^2 + 13}}{2}) \\ 
&= 1 - \left( 2 - \frac{18}{t^2} \right) {\rm sin}^2(\frac{t}{2}) \\
\end{array}
$$

where $ t = \sqrt{\alpha^2 + 13} $.

Now the analytical derivative can be given by:

$$
\begin{array}{ll}
\frac{\mathrm{d} E_\alpha(\alpha)}{\mathrm{d} \alpha} 
&= \left( \frac{18}{t^2} - 2 \right)' {\rm sin}^2(\frac{t}{2}) + \left( \frac{18}{t^2} - 2 \right) \left[ {\rm sin}^2(\frac{t}{2}) \right]' \\
&= -\frac{36}{t^3} \frac{\mathrm{d} t}{\mathrm{d} \alpha} {\rm sin}^2(\frac{t}{2}) + \left( \frac{18}{t^2} - 2 \right) {\rm sin}(\frac{t}{2}) {\rm cos}(\frac{t}{2}) \frac{\mathrm{d} t}{\mathrm{d} \alpha} \\
&= -\frac{36 a}{t^4} {\rm sin}^2(\frac{t}{2}) + \left( \frac{18}{t^2} - 2 \right) {\rm sin}(\frac{t}{2}) {\rm cos}(\frac{t}{2}) \frac{a}{t} \\
\end{array}
$$

When given $ alpha = 1 $, the derivative is $ -0.11372903169365889 $ 🎉
