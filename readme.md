# Relaxations mono neurones renforcées pour la vérification des réseaux neuronaux
**BEKKAR Zakaria, CHRIMNI Walid**

Ceci est une explication et implémentation du papier [The Convex Relaxation Barrier, Revisited: Tightened Single-Neuron Relaxations for Neural Network Verification](https://papers.nips.cc/paper/2020/file/f6c2a0c4b566bc99d596e58638e342b0-Paper.pdf).

# Notations


* $f: \mathbb{R}^{m} \rightarrow \mathbb{R}^{r}$ est un réseau de neurone à une couche composé de N neurones linéairement réparties. Les $m$ premiers neurones sont les neurones d'input, tandis que les N-m neurones restant composent la couche caché. On indexe ces derniers $i=m+1,..., N$
* $w$ les poids.
* $b$ le biais.
* $z_j = x_j \quad \quad \forall j=1, \ldots, m$
* **$\hat{z}$ la variable avant la fonction d'activation** : $$
\hat{z}_{i}=\sum_{j=1}^{i-1} w_{i, j} z_{j}+b_{i} \quad \forall i=m+1, \ldots N, \forall x \in \mathbb{R}^{m}$$
* **$z$ la variable après la fonction d'activation** : $$
z_{i}=\sigma\left(\hat{z}_{i}\right) \quad \forall i=m+1, \ldots, N $$ avec $\sigma$ la fonction ReLU
* **y l'output** : $$
y_{i}=\sum_{j=1}^{N} w_{i, j} z_{j}+b_{i} \quad \forall i=N+1, \ldots, N+r $$

On considère tous ces paramètres (en particulier $w$ et $b$) fixés.

* $[\![ n ]\!] \stackrel{\text { def }}{=}\{1, \ldots, n\}$

* $
\breve{L}_{i} \stackrel{\text { def }}{=}\left\{\begin{array} { l l }  { L _ { i } } & { w _ { i } \geqslant 0 } \\ { U _ { i } } & { \text { sinon } } \end{array} \text { et } \breve { U } _ { i } \stackrel { \text { def } } { = } \left\{\begin{array}{ll} U_{i} & w_{i} \geqslant 0 \\ L_{i} & \text { sinon } \end{array} \text { pour tout } i \in [\![ n ]\!]\right.\right. $

* $\ell(I) \stackrel{\text { def }}{=}  \sum_{i \in I} w_{i} \breve{L}_{i}+\sum_{i \notin I} w_{i} \breve{U}_{i}+b $


* $\mathcal{J} \stackrel{\text { def }}{=}\left\{(I, h) \in 2^{[\![ n ]\!]} \times [\![ n ]\!] \mid \ell(I) \geqslant 0, \quad \ell(I \cup\{h\})<0, \quad w_{i} \neq 0 \forall i \in I\right\}$

* $v(x) \stackrel{\text { def }}{=} \min \left\{\sum_{i \in I} w_{i}\left(x_{i}-\breve{L}_{i}\right)+\frac{\ell(I)}{\breve{U}_{h}-\breve{L}_{h}}\left(x_{h}-\breve{L}_{h}\right) \mid(I, h) \in \mathcal{J}\right\}$

## Introduction

Les attaques adversariales sont un enjeu de fiabilité des réseaux de neuronnes profonds, notamment dans un contexte où ces derniers sont de plus en plus diffusés dans tous les domaines, en particulier des domaines sensibles (médical, social, politique...).

Les attaques adversariales sont des *inputs* de modèles d'apprentissage automatique qu'une entité malveillante a conçus pour que le modèle fasse une erreur. Cette erreur prend la forme d'une classification eronnée d'un input pour lequel une variation imperceptible humainement a été appliquée.  On peut comparer ce procédé à une illusion d'optique car en général, l'humain ne peut pas differencier entre l'input réel et l'attaque adversariale.

Un exemple d'attaque adversariale est proposé ci-dessous. L'input est une image de pandas à laquelle a été ajouté une perturbation calculée ɛ (le changement est imperceptible à l'oeil nu) qui resulte en une prédiction erronée. 


</p>

Szegedy et al. (Intriguing properties of neural networks, Szegedy, C. et al, arXiv preprint arXiv:1312.6199, 2013) proposent d'ajouter une petite perturbation ϵ qui trompe le réseau de classificateurs f en lui faisant choisir la mauvaise classe c pour $\hat{x}=x+ϵ$

\begin{equation}
\underset{ϵ}{\arg \min }|ϵ|_{2}^{2}, \text { s.t } f(x+ϵ)=c, \, \, \, \, \, \, \, x+ϵ \in[0,1]^{n} \,
\end{equation}

$\hat{x}$ est l'exemple le plus proche de x tel que $\hat{x}$ est classé dans la classe c.


Il existe de nombreux types d'attaques adversariales. Goodfellow et al. (Explaining and Harnessing Adversarial Examples, Goodfellow, I.J, Shlens, J. and Szegedy, C., ICLR 2015)
proposent une méthode de principe pour en créer.

### <ins>Construction d'un exemple adversarial classique</ins>

Soient $ x,w,r\in \mathbb{R}^{n}$. On pose $m = \frac{1}{n} \sum_{i=1}^{n}w_i$.

* $x$ correspond à une input d'une couche neuronale
* $w$ correspond au vecteur de poids associés à cette couche
* $r$ correspond à une pertubation que l'on applique à l'input

Étant donné la sortie d'une couche entierement connectée $<w, x>$ on construit un exemple adversarial $\hat{x}$ en y associant une pertubation $r$ telle que :
  $$<w, \hat{x}> \,= \, <w, x> + <w, r>$$.

En particulier, prenons  $r = sign(w)$. Que peut on dire de la statique comparative de $<w,\hat{x}>$ lorsque $n$ augmente?

* On a une variation de cette quantité de $nm$ 

* Cependant, $|r|_∞$ reste constant.

Nous en déduisons que l'ajout d'un petit vecteur r suffit pour perturber significativement l'output $<w,\hat{x}>$.


Dans le sillon de 
Goodfellow et al. (Explaining and Harnessing Adversarial Examples, Goodfellow, I.J, Shlens, J. and Szegedy, C., ICLR 2015) considèrons une linéarisation locale de la perte du réseau
autour de $\theta$ 

$$
\mathcal{L}\left(x_{0}\right) \approx f\left(x_{0}\right)+w \nabla{x} \mathcal{L}\left(\theta, x_{0}, y_{0}\right)
$$

On peut ainsi définir classiquement la pertubation $r$ de la manière suivante : 

$$
\hat{x}=x+\epsilon \operatorname{sign}\left(\nabla_{x} \mathcal{L}(\theta, x, y)\right)
$$

La figure ci-dessous illustre ce procédé : 


Les attaques adversariales sont un aspect de la sécurité sur lequel il est intéressant et urgent de travailler. En effet elles représentent un problème concret de sécurité de l'IA qui peut être traité à court terme. De plus leur résolution est suffisamment complexe qu'elles requirent un effort de recherche important.

## Brève revue de littérature

Le procédé par lequelle la fiabilité d'un modèle est évalué face aux attaques adversariales est appelé test de robustesse.

La recherche dans ce domaine s'est structurée autour de deux approches phares :
* la première, dite exacte (*complete*) : ces algorithmes résolvent exactement le problème, sans erreurs. Ils sont basés sur des techniques dites MIP (*mixed integer programming*) ou SMT (*satisfiability modulo theories*). Les algorithmes de verification exacte se ramènent à la résolution de problèmes NP-hard ce qui limite grandement leur passage à l'échelle et leur mise en pratique.
* la deuxième, dite relaxée (*incomplete*) : 
ces algorithmes se basent historiquement sur des approches de résolutions polynomiales telles que l'optimisation convexe ou la programmation linéaire. Ceci permit le development de méthodes plus efficientes dites *propagation-based*.
La relaxation convexe des contraintes *exactes* qui s'en suit permet un gain de vitesse et de capacité de mise à l'échelle qui se fait au prix d'un précision moindre (augmentation du taux de faux négatifs, i.e du nombre de fois qu'un réseau de neurones n'est pas certifié robuste alors que c'est le cas en réalité.) 

Tout l'enjeux de cet article de Tjandraatmadja et al. est d'optimiser l'arbitrage mis en évidence par les méthodes *incompletes* en proposant un relaxation convexe plus "resserée" i.e plus précise. De ce fait on reste le cadre d'une bonne praticité des algorithmes en terme de vitesse et de passage à l'échelle tout en s'approchant le plus possible du canon théorique des méthodes *exactes*.

Plutôt que de travailler sur un réseau de neurone complet, une simplification classique dont les auteurs tirent partis est de se ramener à un seul neurone. Ceci leur permet d'établir une comparabilité de leur proposition avec l'approche populaire de Δ-relaxation. Celle-ci se fonde sur une relaxation la plus simple et précise possible de la fonction ReLU unavariée. Elle constitue la colonne vertébrale de beaucoup de méthodes de vérification relaxée.

Une limite fondamentale de la Δ-relaxation mise en évidence par Salman et al.[ A convex
relaxation barrier to tight robustness verification of neural networks] est caractérisée par la *barrière de relaxation convexe*. En retour, cet obstacle contraint sévèrement   l'efficacité des méthodes basées sur cette approche comme cela a été démontré de manière computationelle par ces auteurs.

Plusieurs stratégies existent pour dépasser cet obstacle, notamment le fait d'effectuer une relaxation sur plusieurs neurones simultanément. L'ensemble de ces coutournements se font au détriment de la rapidité et de la simplicité. 








Le présent papier propose une amélioration de ces contournements en se basant non pas sur l'espace univarité de la fonction ReLU, mais plutôt sur l'espace affine multivarié de la fonction de pré-activation précédant la fonction ReLU. Des bornes sur chaque neurone sont créees individuellement, puis à travers un algorithme, elles sont utilisés pour donner une borne supérieur sur le problème relaxé.

On peut noter trois contributions principales pour ce papier :

* L'écriture d'une inegalité linéaire pour la relaxation convexe la plus resserée possible. Cette égalité, inspiré de la Δ-relaxation, est plus forte que cette dernière et permet de dépasser la barrière convexe.
* La présentation d'un algorithme qui, étant donné un point, certifie, en temps linéaire, si ce point appartient appartient à l'ensemble de relaxation. Deux algorithmes de verifications sont crée à partir de ce procédé : OptC2V (qui utilise la puissance de la nouvelle relaxation à son plein potentiel), FastC2V (plus rapide, généralisant d'autre algorithme via la nouvelle relaxation)
* Des simulations qui montrent les améliorations importantes que la nouvelle relaxation entraine.

Le présent document se découpe en plusieurs parties :

* Comment dépasser la barrière convexe
* Obtention de bornes adéquates pour un neurone unique 
* Présentation d'un algorithme pour l'obtention de bornes sur le réseau entier
* Algorithme dynamique final : FastC2V






# Dépassement de la barrière convexe

Théoriquement, un réseau de neurone est dit robuste (dans le sens où il passe le test de vérification avec succès) si, étant donné :

* une constante $c \in \mathbb{R}^{r}$
* un polyhèdre $X \subseteq \mathbb{R}^{m}$
*$
\gamma(c, X) \stackrel{\text { def }}{=} \max _{x \in X} c \cdot f(x) \equiv \max _{x, y, \hat{z}, z}\{c \cdot y \mid x \in X\}$
* $\beta \in \mathbb{R}$

on a $\gamma(c,X) \leq \beta $. 


Deux problèmes se posent ici :

* On voudrait pouvoir vérifier cette inégalité pour plusieurs $c$ et $X$ différents afin d'être convaincu que le réseau de neurone est robuste,
* Le problème est NP-hard.

Afin de palier à ces deux problèmes, on "relaxe" le problème, en cherchant un problème dont la fonction objective $\gamma_{R}$ vérifie $\gamma(c,X) \leq \gamma_{R}(c,X)$. Ainsi, on aura $\gamma(c,X) \leq \beta $ dès lors que $\gamma(c,X) \leq \gamma_{R}(c,X)$ et $\gamma_{R}(c,X) \leq \beta$. Il faut avoir la relaxation la plus resserré possible puisqu'on pourrait avoir $
\left.\gamma(c, X) \leqslant \beta<\gamma_{R}(c, X)\right)
$ et ainsi ne pas vérifier un problème qui pourrait l'être.

La majorité des méthodes de relaxation pour ce problème de vérification sont basés sur les variables post-activation. L'approche de ce papier est différente, et c'est ce qui fait sa force. Plutôt que de travailler sur les variables post-activation, les auteurs ont travaillés sur les variables de pré-activation et ont capturés (et relaxés) la non-linéarité introduite par la fonction ReLU.  Pour ce faire, l'ensemble $S^{i} \stackrel{\text { def }}{=}\left\{z \in \mathbb{R}^{i} \mid L \leqslant z_{1: i-1} \leqslant U, \quad z_{i}=\sigma\left(\sum_{j=1}^{i-1} w_{i, j} z_{j}+b_{i}\right)\right\}$ est utilisé, où

* $
z_{1: i-1} \stackrel{\text { def }}{=}\left(z_{1}, \ldots, z_{i-1}\right)
$
* $
L, U \in \mathbb{R}^{i-1}
$ tels que $
L_{j} \leqslant z_{j} \leqslant U_{j}
$

L'ensemble $S^{i}$ est inspiré de la Δ-relaxation
Ainsi, on peut écrire la relaxation de la manière suivante :

$$\gamma_{\mathrm{Elide}}(c, X) \stackrel{\text { def }}{=} \max _{x, y, z}\left\{c \cdot y \mid x \in X,  \quad z_{1: i} \in C_{\mathrm{Elide}}^{i} \forall i=m+1, \ldots, N, \quad y_{i}=\sum_{j=1}^{N} w_{i, j} z_{j}+b_{i} \quad \forall i=N+1, \ldots, N+r\right\} $$ où $
C_{\mathrm{Elide}}^{i} \stackrel{\text { def }}{=} \operatorname{Conv}\left(S^{i}\right) $ l'enveloppe convexe de $S^{i}$.

Cette relaxation permet de dépasser la barrière convexe.

## Des bornes adéquates pour un neurone unique

Afin de construire l'algorithme de relaxation, il convient d'abord de partir d'un unique neuronne puis de l'adapter pour une réseau de neurone. Pour ce faire, une famille d'inégalité est créee puis utilisée pour créer des bornes valides pour chaque neurone successivement.

Etant donnée qu'on se restreint à un seul neurone, on peut donc récrire l'ensemble $S^{i}$ précédent de la manière suivante :

$$
S \stackrel{\text { def }}{=}\{(x, y) \in[L, U] \times \mathbb{R} \mid y=\sigma(f(x))\} $$ où $R,L \in \mathbb{R}$.

Le théorème fondamental sur lequel la majorité des résultats se reposent est le suivant :

**Theoreme 1** :  Si $\ell([\![ n ]\!]) \geqslant 0$, alors $\operatorname{Conv}(S)=S=\{(x, y) \in[L, U] \times \mathbb{R} \mid y=f(x)\} .$ Alternativement, si $\ell(\varnothing)<0$, alors $\operatorname{Conv}(S)=S=[L, U] \times\{0\} .$ Autrement, $\operatorname{Conv}(S)$ est égal à l'ensemble des $(x, y) \in \mathbb{R}^{n} \times \mathbb{R}$ satisfaisant \\

$$
\begin{aligned} • \quad
&y \geqslant w \cdot x+b, \quad y \geqslant 0, \quad L \leqslant x \leqslant U \\
• \quad &y \leqslant \sum_{i \in I} w_{i}\left(x_{i}-\breve{L}_{i}\right)+\frac{\ell(I)}{\breve{U}_{h}-\breve{L}_{h}}\left(x_{h}-\breve{L}_{h}\right) \quad \forall(I, h) \in \mathcal{J} .
\end{aligned} \quad \quad \text{(*)}$$
De plus, si $d \stackrel{\text { def }}{=}\left|\left\{i \in [\![ n ]\!] \mid w_{i} \neq 0\right\}\right|$, alors $d \leqslant|\mathcal{J}| \leqslant\left\lceil\frac{1}{2} d\right\rceil\left(\begin{array}{c}d \\ {\left[\frac{1}{2} d\right]}\end{array}\right)$ et pour chacune de ces inégalités (et pour tout $d \in [\![ n ]\!]$ ) il existe des données qui satisfont l'égalité.


Ce théorème permet de résoudre le problème de séparation très facilement. Pour vérifier si $(x,y) \in \text{Conv}(S)$ on utilise le *théorème de séparation* :

* On vérifie si $(x,y)$ vérifie la première ingéalité du théorème
  * Si oui : si $y \leq v(x)$ alors $(x,y) \in \text{Conv}(S)$ où $v(x) \stackrel{\text { def }}{=} \min \left\{\sum_{i \in I} w_{i}\left(x_{i}-\breve{L}_{i}\right)+\frac{\ell(I)}{\breve{U}_{h}-\breve{L}_{h}}\left(x_{h}-\breve{L}_{h}\right) \mid(I, h) \in \mathcal{J}\right\}$
  * Si non : une solution optimale de $v(x)$ donne $(I, h) \in \mathcal{J}$ qui viole la deuxième inégalité du théorème 1

Ceci avec l'algorithme de l'ellipsoide, nous permet de résoudre efficacement (dans un temps raisonnable) $\gamma_{Elide}$ pour un seul neurone

Ce théorème nous permet d'obtenir des bornes supérieurs pour chaque neurone individuellement (inégalité (\*)). Cependant, il peut avoir un nombre exponentiel d'inégalité (\*) pour chaque neuronne. A priori, on ne sait pas laquelle choisir.

# Un algorithme de base pour la génération de bornes sur le réseau entier

Le papier se base sur un algorithme basé sur la propagation pour générer des bornes fortes pour un réseau de neurone. Cet algorithme est souvent utilisé comme base par plusieurs algorithmes de la littérature. Il est également utilisé ici comme base afin de produire une borne supérieure plus resserrée.

Soit $\mathcal{C}(z)=\sum_{i=1}^{\eta} c_{i} z_{i}+b \text { pour } \eta \leqslant N$ et $X \subseteq \mathbb{R}^{m}$ un ensemble borné. Le but est de créer un algorithme permettant de produire une borne supérieur valide pour $\mathcal{C}$.

Tout d'abord, nous allons construire une famille de fonctions affines $\left\{\mathcal{L}_{i}, \mathcal{U}_{i}\right\}_{i=m+1}^{\eta}$ tels que 

$$
\mathcal{L}_{i}\left(z_{1: i-1}\right) \leqslant z_{i} \leqslant \mathcal{U}_{i}\left(z_{1: i-1}\right) \quad \forall i=m+1, \ldots, \eta
$$

Pour ce faire, nous allons utiliser des *scalaires* $\hat{L}_{i}, \hat{U}_{i} \in \mathbb{R}$ qui bornent la variable de pré-activation $̂\hat{z}_{i}$ : $\hat{L}_{i} \leq \hat{z}_{i} \leq \hat{U}_{i}$. On obtient ces scalaires successivement pour $i=1,...\eta$ : étant donné que $\hat{z}_{i} = \sum_{j=1}^{i-1} w_{i, j} z_{j}+b_{i}$, on construit directement $\hat{L}_{m+1}, \hat{U}_{m+1}$, puis $\hat{L}_{m+2}, \hat{U}_{m+2}$ etc... On peut alors écrire :

* $\mathcal{L}_{i}(z_{1:i-1}) = \begin{cases} \sum_{j=1}^{i-1}  w_{i, j} z_{j}+b_{i} & \text{si } \hat{U}_{i} \leq 0 \\ 0 & \text {si } \hat{L}_{i} \geq 0 \\
\frac{\hat{U}_{i}}{\hat{U}_{i}-\hat{L}_{i}}\left(\sum_{j=1}^{i-1} w_{i, j} z_{j}+b_{i}\right) & \text{sinon }\end{cases}$

* $\mathcal{U}_{i}(z_{1:i-1}) = \begin{cases} \sum_{j=1}^{i-1}  w_{i, j} z_{j}+b_{i} & \text{si } \hat{U}_{i} \leq 0 \\ 0 & \text {si } \hat{L}_{i} \geq 0 \\
\frac{\hat{U}_{i}}{\hat{U}_{i}-\hat{L}_{i}}\left(\sum_{j=1}^{i-1} w_{i, j} z_{j}+b_{i}-\hat{L}_{i}\right) & \text{sinon }\end{cases}$.

Ce sont ces bornes qui sont utilisés dans la plupart des algorithmes de verification. Dans ce papier, l'approche est différente. Les auteurs proposent d'utiliser une inégalité (\*) du théorème 1 comme borne supérieur. Etant donné qu'il existe un nombre exponentiel d'inégalité (\*), on ne sait pas a priori laquelle choisir pour appliquer l'algorithme. La solution trouvée à ce problème est de répeter l'algorithme plusieurs fois avec différentes bornes supérieures et de prendre, parmis toutes les itérations de l'algorithmes effectués, la meilleur borne.

Via ces bornes, nous pouvons construire le **problème d'optimisation** suivant :

$$ \text{Probleme d'optimisation (P) : }
\begin{array}{rl}
B(\mathcal{C}, \eta) \stackrel{\text { def }}{=} \max _{z} & \mathcal{C}(z) \equiv \sum_{i=1}^{\eta} c_{i} z_{i}+b \\
\text { s.t. } & z_{1: m} \in X \\
& \mathcal{L}_{i}\left(z_{1: i-1}\right) \leqslant z_{i} \leqslant \mathcal{U}_{i}\left(z_{1: i-1}\right) \quad \forall i=m+1, \ldots, \eta
\end{array}
$$


de sorte que $B(\mathcal{C}, \eta) = \max _{x \in X} \mathcal{C}\left(z_{1: \eta}(x)\right)$, ce qui nous permet, en résolvant (P), de récuperer la valeur de $\max _{x \in X} \mathcal{C}\left(z_{1: \eta}(x)\right)$. 

### Résolution via le backward pass

$B(\mathcal{C}, \eta)$ est obtenu en utilisant le **backward pass** et la méthode d'élimination de Fourier-Motzkin : on remplace successivement les $z_i$ dans $\mathcal{C}$ par la borne associée qui sature la contrainte. Cette méthode nous donne cepandant seulement la valeur optimale valeur optimale $B(\mathcal{C}, \eta)$ et une **solution partielle** $z_{1:m}$


### Solution complète via le forward pass

Cependant, cet algorithme nous permet seulement d'avoir la valeur optimale $B(\mathcal{C}, \eta)$ et une solution partielle $z_{1:m}$. On obtient la solution complète $z^{*}_{1:n}$ de la manière suivante :

* $z^{*}_{1:m} = z_{1:m}$  avec $z_{1:m} $la solution partielle
* $\forall i=m+1,\ldots,\eta \quad \ z^*_i = \begin{cases} \mathcal{U}(z^*_{1:i-1}) & \text{si c'est la borne supérieur i qui était saturée dans le backward pass }\\ \mathcal{L}(z^*_{1:i-1}) & \text {si c'est la borne supérieur i qui était saturée dans le backward pass}\end{cases}$



Cet étape est appelé le forward pass.

La solution complète $z^*_{1:N}$ est indispensable afin de mettre en place l'algorithme final.





# Algorithme dynamique final : FastC2V



Toutes les étapes précédentes ont servit à la préparation de l'algorithme dynamique suivant que les chercheurs de ce papier ont mis en place. Le but est de produire la borne supérieur la plus resserré possible d'une fonction affine. L'algorithme est par la suite utilisé sur la fonction objectif de $γ_{Elide}$ (le problème relaxé). Cet algorithme revient à approcher la valeur de $\gamma_{Elide}$ :

* Etant donné :
  * un ensemble $X \subseteq \mathbb{R}^{m}$
  * une fonction affine $\mathcal{C}: \mathbb{R}^{\eta} \rightarrow \mathbb{R}$
  * des fonctions affines qui bornent "l'élément" $i$ de $\mathcal{C} :$   $\left\{\mathcal{L}_{i}, \mathcal{U}_{i}\right\}_{i=m+1}^{\eta}$
  * un nombre d'itération T

l'algorithme renvoie une borne supérieur de $\max _{x \in X} \mathcal{C}(z_{1:\eta}(x))$

L'algorithme se présente comme suit :

* On effecute le backward passe afin de récuperer $z^*_{1:m}$ une solution optimale partielle et $B_0 = \mathcal{C}(z^*_{1:m})$
* On effectue l'étape suivante pour $j=1,\ldots,T$ :
  * On récupère la solution complète $z^*_{1:\eta}$ par forward pass
  * pour $i=m+1,\ldots,\eta$ :
    * on note $\mathcal{U_i'}$ l'inégalité (\*) la plus violée par $z^*_{1:\eta}$ et $v$ sa violation
    * si $v \geq 0$ alors on actualise la valeur de $\mathcal{U}_i$ par $\mathcal{U_i'}$
  * On effectue de nouveau le backward pass avec les nouveaux $\mathcal{U_i'}$ et on stocke la valeur de  dans $B_j=\mathcal{C}(z_{1:\eta}^*(x))$
* Enfin, on retourne $min_{j=0,\ldots,T} B_j$


# Simulations

Deux méthodes ont été évalués par les chercheurs : l'algorithme de la question précédente (FastC2V)) et une méthode résolvant partiellement le problème de programmation linéaire du théorème 1 en traitant les inégalités (\*) comme des plants sécants (OptC2V). La structure générale est la même pour les deux méthodes : les bornes scalaires pour les variables de pré-activation sont calculés pour chaque neurone au fur et à mesure que nous avançons dans le réseau, puis ces bornes sont utilisés pour produire les fonctions affines bornantes. Pour chaque neurone, les bornes sclaires sont créees de la manière suivante :

* pour FastC2V : il s'agit de l'algorithme de la partie précédente avec  $\left\{\mathcal{L}_{i}, \mathcal{U}_{i}\right\}_{i=m+1}^{\eta}$ provenant d'algorithmes annexes nommés DeepPoly et CROWN-Ada.
* pour OptC2V : chaque borne est générée en résolvant une série de problème de programmation linéaire où  les inégalités de borne supérieure sont générées dynamiquement et ajoutées en tant que plans sécants

Chaque méthode est comparé à ses méthodes "baseline" naturelle : DeepPoly pour FastC2V et la Δ-relaxation pour OptC2V.


Le problème de vérification est le suivant : 

étant donnée : 
* une image $\hat{x} \in [0,1]^m$ labélisé $t$
* un réseau de neurone où $f_k(x)$ renvoie le logit pour la classe k
* une distance $\epsilon$

l'image $\hat{x}$ est vérifiée robuste si : $\max _{x \in[\hat{L}, \hat{U}]} \max _{k \in K}\left\{f_{k}(x)-f_{t}(x)\right\}<0$, où $\hat{L}_{i}=\max \left\{0, \hat{x}_{i}-\epsilon\right\}$ et $\hat{U}_{i}=\min \left\{1, \hat{x}_{i}+\epsilon\right\}$ pour tout $i=1, \ldots, m$.

Les datasets utilisés sont MNIST et CIFAR-10 (avec plusieurs variations à chaque fois) et plusieurs distances $\epsilon$ sont utilisées.

## Résultats

$$
\begin{array}{ll|r|r|r|r|r|r|r|r}
\text { Method } & & \text{MNIST } 6 \times 100 & \text{MNIST } 9 \times 100 & \text{MNIST } 6 \times 200 & \text{MNIST } 9 \times 200 & \text{MNIST } \text { ConvS } & \text{MNIST } \text { ConvB } & \text{MNIST } \text { ConvS } \\
\hline {\text { DeepPoly }} & \text { #verified } & 160 & 182 & 292 & 259 & 162 & 652 & 359 \\
& \text { Time (s) } & 0.7 & 1.4 & 2.4 & 5.6 & 0.9 & 7.4 & 2.8 \\
\text { FastC2V } & \text { #verified } & 279 & 269 & 477 & 392 & 274 & 691 & 390 \\
& \text { Time (s) } & 8.7 & 19.3 & 25.2 & 57.2 & 5.3 & 16.3 & 15.3 \\
\text { LP } & \text { #verified } & 201 & 223 & 344 & 307 & 242 & 743 & 373 \\
& \text { Time (s) } & 50.5 & 385.6 & 218.2 & 2824.7 & 23.1 & 24.9 & 38.1 \\
\text { OptC2V } & \text { #verified } & 429 & 384 & 601 & 528 & 436 & 771 & 398 \\
& \text { Time (s) } & 136.7 & 759.4 & 402.8 & 3450.7 & 55.4 & 102.0 & 104.8 \\
\hline \text { RefineZono } & \text { #verified } & 312 & 304 & 341 & 316 & 179 & 648 & 347 \\
\text { kPoly } & \text { #verified } & 441 & 369 & 574 & 506 & 347 & 736 & 399 \\
\hline \text { Upper bound } & \text { #verified } & 842 & 820 & 901 & 911 & 746 & 831 & 482 \\
\hline
\end{array}
$$

Bien que FastC2V est un plus lent que DeepPoly, il reste relativement rapide et surtout il permet de vérifier un nombre plus élevé d'images. On retrouve le même constat entre LP et OptC2V.

On note même que ces deux méthodes sont parfois meilleur que RefineZono, une méthode fortement fine-tuné qui combine LP et DeepPoly. Le fine-tuning et le calcul complexe de RefineZero étant très couteux, les résultats obtenus sont très prometeur. 


# Critiques 

Le papier de recherche atteint bien son but et permet en effet d'outrepasser la barrière convexe. Cepandant, nous avons quelques critiques à énoncer :

* Le papier est également difficile à comprendre dans la façon dont il est écrit. Nous concevons que c'est un papier de recherche et que, par conséquent, nous ne sommes pas tenus par la main pour le comprendre, mais certaines étapes énoncés dans le papier restent assez obscure et peu détaillés. Les auteurs attendent que l'on devinent certaines choses. Par exemple, l'abstract est assez obscure.
* Le papier est assez difficile à comprendre dans sa structure. Il nous a fallu beaucoup de temps pour en comprendre les tenants et les aboutissants. La structure est assez floue, on ne comprend la contribution finale des chercheurs qu'au dernier paragraphe (hors paragraphe simulation). Une sorte de résumé est bien disponible au début, mais elle nous parait assez flou. Par exemple, on aurait gagné en clarté si la partie 4.2 (où l'on apprend comment créer des bornes pour l'input de l'algorithme de la partie 4.1) avait été placé avant la présentation de l'algorithme (dans la partie 4.1)
* En voulant généraliser, nous trouvons que le papier se complique sans apporter plus d'éléments pertinents.
* La partie code est très obscure. Le repertoire github ne contient que très peu, voire aucune information sur le code (voir https://github.com/google-research/tf-opt). Nous ne savons pas à quoi correspond chaque partie et nous sommes un peu livré à nous même concernant le lancement du code. En particulier, nous n'avons pas pu lancé le code dû à une incompatibilité de compilateur, et aucune information n'est disponible à ce sujet. De ce fait, il nous est impossible de reproduire les résultats des simulations.

Malgré cela, nous notons que l'algorithme mis en place est très efficient (comme en témoigne la partie précédente) ce qui fait que ce papier est  une réussite. Les auteurs sont également très au courant du fait que leur algorithme ne certifie la robustesse qu'en un certain sens et que leur algorithme ne fonctionnerait pas aussi bien si on définissait la robustesse différemment. Ils sont conscient des avantages et défauts de leur production.

# Parallèle empirique avec une méthode de vérification connexe




Une unité algorithmique fondamentale de *FastC2V* est l'algorithme de propagation. D'autres méthodes de vérification se base également sur cette brique afin de dérouler leurs approches.

En particulier, les travaux de Singh, Gagandeep and Gehr, Timon and Mirman, Matthew and Püschel, Markus and Vechev, Martin dans le cadre du papier *Fast and Effective Robustness Certification* ont influencé le domaine. Notamment, il a été cités plus de 250 fois, en particulier dans le présent papier. 

Ces aspects là ainsi que la particularité des outils mathématiques employés nous ont aménés à nous interesser à *DeepZ* ainsi qu'à produire un parallèle empirique avec *FastC2V*.


## Certification de la robustesse d'un réseau de neurones en utilisant la notion de zonotope via la méthode DeepZ

Papier original : https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf

### Considerations théoriques

La méthode *DeepZ* permet de certifier la robustesse de réseaux de neurones en se basant sur des *interpretations abstraites*. Cette méthode fait donc partie des approches dites incomplètes ou rélaxées.

L'idée générale est d'approximer le comportement du réseau, via le formalisme de *l'interpretation abstraite*. *DeepZ* tire partie  du *Zonotope Abstract Domain* afin d'obtenir un moyen précis et efficient de capturer l'effet des transformations affines à l'intérieur du réseau.


La zonotope abstraction utilisée par le papier original est la suivante:

\begin{equation}
    \hat{x} = \eta_0 + \sum_{i=1}^{i=N} \eta_i \epsilon_i 
\end{equation}

où $\eta_0$ le vecteur central, $\epsilon_i$ representent le bruit, $\eta_i$ les coefficients de deviations autour de $\eta_0$.


We can illustrate a 2D toy example of this below in which the initial datapoint has two features, with a central vector of [0.25, 0.25] and these features both have noise terms of [0.25, 0.25]. We push this zonotope through the neural network and show it's intermediate shapes:

On peut illustrer à l'aide d'un petit exemple l'impact de l'application d'une couche neuronale sur le zonotope dérivé de la zonotope abstraction ci-dessus.



Nous pouvons voir que le zonotope change au fur et à mesure de l'application des différentes parties de la couche. La première transformation linéaire n'ahoute pas d'ensemble de ligne parrallèle alors que la dernière transformation ReLU ajoute un troisième ensemble de lignes parrallèle.

### Application sur MNIST

Dans le cadre de cette application sur le dataset MNIST, nous allons utiliser l'implementation art de *DeepZ*


```python
!pip install adversarial-robustness-toolbox
```

    Collecting adversarial-robustness-toolbox
      Downloading adversarial_robustness_toolbox-1.10.0-py3-none-any.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 5.4 MB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from adversarial-robustness-toolbox) (1.15.0)
    Requirement already satisfied: numpy>=1.18.0 in /usr/local/lib/python3.7/dist-packages (from adversarial-robustness-toolbox) (1.21.5)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from adversarial-robustness-toolbox) (57.4.0)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from adversarial-robustness-toolbox) (1.4.1)
    Requirement already satisfied: scikit-learn<1.1.0,>=0.22.2 in /usr/local/lib/python3.7/dist-packages (from adversarial-robustness-toolbox) (1.0.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from adversarial-robustness-toolbox) (4.63.0)
    Collecting numba>=0.53.1
      Downloading numba-0.55.1-1-cp37-cp37m-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.3 MB)
    [K     |████████████████████████████████| 3.3 MB 31.1 MB/s 
    [?25hCollecting llvmlite<0.39,>=0.38.0rc1
      Downloading llvmlite-0.38.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.5 MB)
    [K     |████████████████████████████████| 34.5 MB 12 kB/s 
    [?25hRequirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn<1.1.0,>=0.22.2->adversarial-robustness-toolbox) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn<1.1.0,>=0.22.2->adversarial-robustness-toolbox) (3.1.0)
    Installing collected packages: llvmlite, numba, adversarial-robustness-toolbox
      Attempting uninstall: llvmlite
        Found existing installation: llvmlite 0.34.0
        Uninstalling llvmlite-0.34.0:
          Successfully uninstalled llvmlite-0.34.0
      Attempting uninstall: numba
        Found existing installation: numba 0.51.2
        Uninstalling numba-0.51.2:
          Successfully uninstalled numba-0.51.2
    Successfully installed adversarial-robustness-toolbox-1.10.0 llvmlite-0.38.0 numba-0.55.1





```python
import torch
import torch.optim as optim
import numpy as np

from torch import nn
from sklearn.utils import shuffle

from art.estimators.certification import deep_z
from art.utils import load_mnist, preprocess, to_categorical

device = 'cpu'
```

Nous definissons ici le réseau de neurones classifiant les différents chiffres de MNIST


```python
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               dilation=(1, 1),
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               dilation=(1, 1),
                               padding=(0, 0))
        self.fc1 = nn.Linear(in_features=800,
                             out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
```


```python
model = MNISTModel()
opt = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

x_test = np.squeeze(x_test)
x_test = torch.tensor(np.expand_dims(x_test, axis=1)).to(device)
y_test = torch.tensor(np.argmax(y_test, axis=1)).to(device)


x_train = np.squeeze(x_train)
x_train = torch.tensor(np.expand_dims(x_train, axis=1)).to(device)
y_train = torch.tensor(np.argmax(y_train, axis=1)).to(device)
```

Entrainement du modèle


```python
def standard_train(model, opt, criterion, x, y, bsize=32, epochs=5):
    num_of_batches = int(len(x) / bsize)
    for epoch in range(epochs):
        x, y = shuffle(x, y)
        loss_list = []
        for bnum in range(num_of_batches):
            x_batch = np.copy(x[bnum * bsize:(bnum + 1) * bsize].to('cpu'))
            y_batch = np.copy(y[bnum * bsize:(bnum + 1) * bsize].to('cpu'))

            x_batch = torch.from_numpy(x_batch).float().to('cpu')
            y_batch = torch.from_numpy(y_batch).type(torch.LongTensor).to('cpu')

            # zero the parameter gradients
            opt.zero_grad()
            outputs = model(x_batch.to('cpu'))
            loss = criterion(outputs, y_batch)
            loss_list.append(loss.data)
            loss.backward()
            opt.step()
        print('End of epoch {} loss {}'.format(epoch, np.mean(loss_list)))
    return model

model = standard_train(model=model,
                       opt=opt,
                       criterion=criterion,
                       x=x_train,
                       y=y_train)

```

    End of epoch 0 loss 0.5703016519546509
    End of epoch 1 loss 0.2836313843727112
    End of epoch 2 loss 0.20428431034088135
    End of epoch 3 loss 0.1470993310213089
    End of epoch 4 loss 0.11293710023164749


Evaluation purement empirique du modèle


```python
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
x_test = np.squeeze(x_test)
x_test = np.expand_dims(x_test, axis=1)
```


```python
with torch.no_grad():
    test_preds = model(torch.from_numpy(x_test).float().to(device))

test_preds = np.argmax(test_preds.cpu().detach().numpy(), axis=1)
print('Test acc: ', np.mean(test_preds == y_test) * 100)
```

    Test acc:  97.25


Toute la problématique est de savoir à quel point cette métrique est robsute. Nous pouvons désormais commencer à examiner la robustesse certifée de ce réseau de neurones.


```python
zonotope_model = deep_z.PytorchDeepZ(model=model, 
                                     clip_values=(0, 1), 
                                     loss=nn.CrossEntropyLoss(), 
                                     input_shape=(1, 28, 28), 
                                     nb_classes=10)
```

    registered <class 'torch.nn.modules.conv.Conv2d'>
    registered <class 'torch.nn.modules.activation.ReLU'>
    registered <class 'torch.nn.modules.conv.Conv2d'>
    registered <class 'torch.nn.modules.activation.ReLU'>
    registered <class 'torch.nn.modules.linear.Linear'>
    Inferred reshape on op num 4


    /usr/local/lib/python3.7/dist-packages/art/estimators/certification/deep_z/pytorch.py:89: UserWarning: 
    This estimator does not support networks which have dense layers before convolutional. We currently infer a reshape when a neural network goes from convolutional layers to dense layers. If your use case does not fall into this pattern then consider directly building a certifier network with the custom layers found in art.estimators.certification.deepz.deep_z.py
    
      "\nThis estimator does not support networks which have dense layers before convolutional. "


A la suite du papier, nous devons définir la borne à vérifier.
Ici nous choisissons de verifier la robustesse $L_{∞}$ avec une borne de $0.05$ sur $100$ images


```python
from google.colab.output import eval_js
eval_js('google.colab.output.setIframeHeight("500")')
bound = 0.05
num_certified = 0
num_correct = 0
sample_size = 1000

original_x = np.copy(x_test)
for i, (sample, pred, label) in enumerate(zip(x_test[:sample_size], test_preds[:sample_size], y_test[:sample_size])):
    
    eps_bound = np.eye(784) * bound
    sample, eps_bound = zonotope_model.pre_process(cent=sample,eps=eps_bound)
    sample = np.expand_dims(sample, axis=0)
    is_certified = zonotope_model.certify(cent=sample,
                                          eps=eps_bound,
                                          prediction=pred)
    if pred == label:
        num_correct +=1
        if is_certified:
            num_certified +=1
    print('Classifié correctement {}/{} et certifié {}/{}'.format(num_correct, i+1, num_certified, i+1))
```

    Classifié correctement 1/1 et certifié 1/1
    Classifié correctement 2/2 et certifié 2/2
    Classifié correctement 3/3 et certifié 2/3
    Classifié correctement 4/4 et certifié 3/4
    Classifié correctement 5/5 et certifié 4/5
    Classifié correctement 6/6 et certifié 4/6
    Classifié correctement 7/7 et certifié 4/7
    Classifié correctement 8/8 et certifié 4/8
    Classifié correctement 8/9 et certifié 4/9
    Classifié correctement 9/10 et certifié 5/10
    Classifié correctement 10/11 et certifié 6/11
    Classifié correctement 11/12 et certifié 7/12
    Classifié correctement 12/13 et certifié 8/13
    Classifié correctement 13/14 et certifié 9/14
    Classifié correctement 14/15 et certifié 10/15
    Classifié correctement 15/16 et certifié 11/16
    Classifié correctement 16/17 et certifié 12/17
    Classifié correctement 17/18 et certifié 13/18
    Classifié correctement 18/19 et certifié 13/19
    Classifié correctement 19/20 et certifié 14/20
    Classifié correctement 20/21 et certifié 14/21
    Classifié correctement 21/22 et certifié 15/22
    Classifié correctement 22/23 et certifié 16/23
    Classifié correctement 23/24 et certifié 17/24
    Classifié correctement 24/25 et certifié 17/25
    Classifié correctement 25/26 et certifié 18/26
    Classifié correctement 26/27 et certifié 18/27
    Classifié correctement 27/28 et certifié 19/28
    Classifié correctement 28/29 et certifié 20/29
    Classifié correctement 29/30 et certifié 20/30
    Classifié correctement 30/31 et certifié 21/31
    Classifié correctement 31/32 et certifié 21/32
    Classifié correctement 32/33 et certifié 22/33
    Classifié correctement 33/34 et certifié 22/34
    Classifié correctement 34/35 et certifié 23/35
    Classifié correctement 35/36 et certifié 24/36
    Classifié correctement 36/37 et certifié 24/37
    Classifié correctement 37/38 et certifié 24/38
    Classifié correctement 38/39 et certifié 24/39
    Classifié correctement 39/40 et certifié 24/40
    Classifié correctement 40/41 et certifié 24/41
    Classifié correctement 41/42 et certifié 24/42
    Classifié correctement 42/43 et certifié 25/43
    Classifié correctement 43/44 et certifié 26/44
    Classifié correctement 44/45 et certifié 26/45
    Classifié correctement 45/46 et certifié 26/46
    Classifié correctement 46/47 et certifié 26/47
    Classifié correctement 47/48 et certifié 27/48
    Classifié correctement 48/49 et certifié 28/49
    Classifié correctement 49/50 et certifié 29/50
    Classifié correctement 50/51 et certifié 30/51
    Classifié correctement 51/52 et certifié 31/52
    Classifié correctement 52/53 et certifié 32/53
    Classifié correctement 53/54 et certifié 33/54
    Classifié correctement 54/55 et certifié 34/55
    Classifié correctement 55/56 et certifié 35/56
    Classifié correctement 56/57 et certifié 36/57
    Classifié correctement 57/58 et certifié 36/58
    Classifié correctement 58/59 et certifié 37/59
    Classifié correctement 59/60 et certifié 37/60
    Classifié correctement 60/61 et certifié 38/61
    Classifié correctement 61/62 et certifié 39/62
    Classifié correctement 62/63 et certifié 39/63
    Classifié correctement 63/64 et certifié 39/64
    Classifié correctement 64/65 et certifié 40/65
    Classifié correctement 65/66 et certifié 40/66
    Classifié correctement 66/67 et certifié 40/67
    Classifié correctement 67/68 et certifié 41/68
    Classifié correctement 68/69 et certifié 42/69
    Classifié correctement 69/70 et certifié 43/70
    Classifié correctement 70/71 et certifié 44/71
    Classifié correctement 71/72 et certifié 45/72
    Classifié correctement 72/73 et certifié 46/73
    Classifié correctement 73/74 et certifié 46/74
    Classifié correctement 74/75 et certifié 46/75
    Classifié correctement 75/76 et certifié 47/76
    Classifié correctement 76/77 et certifié 48/77
    Classifié correctement 77/78 et certifié 49/78
    Classifié correctement 78/79 et certifié 49/79
    Classifié correctement 79/80 et certifié 50/80
    Classifié correctement 80/81 et certifié 50/81
    Classifié correctement 81/82 et certifié 51/82
    Classifié correctement 82/83 et certifié 52/83
    Classifié correctement 83/84 et certifié 52/84
    Classifié correctement 84/85 et certifié 53/85
    Classifié correctement 85/86 et certifié 54/86
    Classifié correctement 86/87 et certifié 55/87
    Classifié correctement 87/88 et certifié 56/88
    Classifié correctement 88/89 et certifié 57/89
    Classifié correctement 89/90 et certifié 57/90
    Classifié correctement 90/91 et certifié 58/91
    Classifié correctement 91/92 et certifié 59/92
    Classifié correctement 92/93 et certifié 59/93
    Classifié correctement 93/94 et certifié 60/94
    Classifié correctement 94/95 et certifié 60/95
    Classifié correctement 95/96 et certifié 61/96
    Classifié correctement 96/97 et certifié 61/97
    Classifié correctement 97/98 et certifié 62/98
    Classifié correctement 98/99 et certifié 63/99
    Classifié correctement 99/100 et certifié 64/100
    Classifié correctement 100/101 et certifié 65/101
    Classifié correctement 101/102 et certifié 66/102
    Classifié correctement 102/103 et certifié 67/103
    Classifié correctement 103/104 et certifié 68/104
    Classifié correctement 104/105 et certifié 68/105
    Classifié correctement 105/106 et certifié 69/106
    Classifié correctement 106/107 et certifié 70/107
    Classifié correctement 107/108 et certifié 70/108
    Classifié correctement 108/109 et certifié 70/109
    Classifié correctement 109/110 et certifié 71/110
    Classifié correctement 110/111 et certifié 72/111
    Classifié correctement 111/112 et certifié 72/112
    Classifié correctement 112/113 et certifié 73/113
    Classifié correctement 113/114 et certifié 74/114
    Classifié correctement 114/115 et certifié 74/115
    Classifié correctement 115/116 et certifié 74/116
    Classifié correctement 116/117 et certifié 74/117
    Classifié correctement 117/118 et certifié 75/118
    Classifié correctement 118/119 et certifié 75/119
    Classifié correctement 119/120 et certifié 76/120
    Classifié correctement 120/121 et certifié 77/121
    Classifié correctement 121/122 et certifié 77/122
    Classifié correctement 122/123 et certifié 77/123
    Classifié correctement 123/124 et certifié 78/124
    Classifié correctement 124/125 et certifié 78/125
    Classifié correctement 125/126 et certifié 78/126
    Classifié correctement 126/127 et certifié 79/127
    Classifié correctement 127/128 et certifié 80/128
    Classifié correctement 128/129 et certifié 81/129
    Classifié correctement 129/130 et certifié 82/130
    Classifié correctement 130/131 et certifié 83/131
    Classifié correctement 131/132 et certifié 84/132
    Classifié correctement 132/133 et certifié 85/133
    Classifié correctement 133/134 et certifié 86/134
    Classifié correctement 134/135 et certifié 87/135
    Classifié correctement 135/136 et certifié 88/136
    Classifié correctement 136/137 et certifié 89/137
    Classifié correctement 137/138 et certifié 90/138
    Classifié correctement 138/139 et certifié 91/139
    Classifié correctement 139/140 et certifié 92/140
    Classifié correctement 140/141 et certifié 93/141
    Classifié correctement 141/142 et certifié 94/142
    Classifié correctement 142/143 et certifié 94/143
    Classifié correctement 143/144 et certifié 94/144
    Classifié correctement 144/145 et certifié 94/145
    Classifié correctement 145/146 et certifié 94/146
    Classifié correctement 146/147 et certifié 95/147
    Classifié correctement 147/148 et certifié 96/148
    Classifié correctement 148/149 et certifié 97/149
    Classifié correctement 148/150 et certifié 97/150
    Classifié correctement 149/151 et certifié 98/151
    Classifié correctement 150/152 et certifié 98/152
    Classifié correctement 151/153 et certifié 99/153
    Classifié correctement 152/154 et certifié 99/154
    Classifié correctement 153/155 et certifié 99/155
    Classifié correctement 154/156 et certifié 100/156
    Classifié correctement 155/157 et certifié 101/157
    Classifié correctement 156/158 et certifié 102/158
    Classifié correctement 157/159 et certifié 102/159
    Classifié correctement 158/160 et certifié 102/160
    Classifié correctement 159/161 et certifié 103/161
    Classifié correctement 160/162 et certifié 104/162
    Classifié correctement 161/163 et certifié 105/163
    Classifié correctement 162/164 et certifié 106/164
    Classifié correctement 163/165 et certifié 107/165
    Classifié correctement 164/166 et certifié 108/166
    Classifié correctement 165/167 et certifié 109/167
    Classifié correctement 166/168 et certifié 109/168
    Classifié correctement 167/169 et certifié 109/169
    Classifié correctement 168/170 et certifié 110/170
    Classifié correctement 169/171 et certifié 111/171
    Classifié correctement 170/172 et certifié 111/172
    Classifié correctement 171/173 et certifié 111/173
    Classifié correctement 172/174 et certifié 112/174
    Classifié correctement 173/175 et certifié 113/175
    Classifié correctement 174/176 et certifié 113/176
    Classifié correctement 175/177 et certifié 113/177
    Classifié correctement 176/178 et certifié 113/178
    Classifié correctement 177/179 et certifié 114/179
    Classifié correctement 178/180 et certifié 115/180
    Classifié correctement 179/181 et certifié 115/181
    Classifié correctement 180/182 et certifié 116/182
    Classifié correctement 181/183 et certifié 116/183
    Classifié correctement 182/184 et certifié 117/184
    Classifié correctement 183/185 et certifié 117/185
    Classifié correctement 184/186 et certifié 117/186
    Classifié correctement 185/187 et certifié 118/187
    Classifié correctement 186/188 et certifié 118/188
    Classifié correctement 187/189 et certifié 119/189
    Classifié correctement 188/190 et certifié 119/190
    Classifié correctement 189/191 et certifié 119/191
    Classifié correctement 190/192 et certifié 119/192
    Classifié correctement 191/193 et certifié 120/193
    Classifié correctement 192/194 et certifié 120/194
    Classifié correctement 193/195 et certifié 121/195
    Classifié correctement 194/196 et certifié 121/196
    Classifié correctement 195/197 et certifié 121/197
    Classifié correctement 196/198 et certifié 122/198
    Classifié correctement 197/199 et certifié 122/199
    Classifié correctement 198/200 et certifié 123/200
    Classifié correctement 199/201 et certifié 124/201
    Classifié correctement 200/202 et certifié 125/202
    Classifié correctement 201/203 et certifié 125/203
    Classifié correctement 202/204 et certifié 125/204
    Classifié correctement 203/205 et certifié 125/205
    Classifié correctement 204/206 et certifié 126/206
    Classifié correctement 205/207 et certifié 126/207
    Classifié correctement 206/208 et certifié 127/208
    Classifié correctement 207/209 et certifié 128/209
    Classifié correctement 208/210 et certifié 128/210
    Classifié correctement 209/211 et certifié 129/211
    Classifié correctement 210/212 et certifié 129/212
    Classifié correctement 211/213 et certifié 130/213
    Classifié correctement 212/214 et certifié 131/214
    Classifié correctement 213/215 et certifié 131/215
    Classifié correctement 214/216 et certifié 132/216
    Classifié correctement 215/217 et certifié 133/217
    Classifié correctement 216/218 et certifié 133/218
    Classifié correctement 217/219 et certifié 134/219
    Classifié correctement 218/220 et certifié 134/220
    Classifié correctement 219/221 et certifié 134/221
    Classifié correctement 220/222 et certifié 135/222
    Classifié correctement 221/223 et certifié 136/223
    Classifié correctement 222/224 et certifié 137/224
    Classifié correctement 223/225 et certifié 137/225
    Classifié correctement 224/226 et certifié 138/226
    Classifié correctement 225/227 et certifié 139/227
    Classifié correctement 226/228 et certifié 139/228
    Classifié correctement 227/229 et certifié 139/229
    Classifié correctement 228/230 et certifié 139/230
    Classifié correctement 229/231 et certifié 140/231
    Classifié correctement 230/232 et certifié 140/232
    Classifié correctement 231/233 et certifié 140/233
    Classifié correctement 232/234 et certifié 140/234
    Classifié correctement 233/235 et certifié 140/235
    Classifié correctement 234/236 et certifié 140/236
    Classifié correctement 235/237 et certifié 141/237
    Classifié correctement 236/238 et certifié 142/238
    Classifié correctement 237/239 et certifié 143/239
    Classifié correctement 238/240 et certifié 144/240
    Classifié correctement 239/241 et certifié 145/241
    Classifié correctement 240/242 et certifié 145/242
    Classifié correctement 241/243 et certifié 146/243
    Classifié correctement 242/244 et certifié 146/244
    Classifié correctement 243/245 et certifié 146/245
    Classifié correctement 244/246 et certifié 146/246
    Classifié correctement 245/247 et certifié 147/247
    Classifié correctement 245/248 et certifié 147/248
    Classifié correctement 246/249 et certifié 148/249
    Classifié correctement 247/250 et certifié 148/250
    Classifié correctement 248/251 et certifié 149/251
    Classifié correctement 249/252 et certifié 149/252
    Classifié correctement 250/253 et certifié 150/253
    Classifié correctement 251/254 et certifié 150/254
    Classifié correctement 252/255 et certifié 151/255
    Classifié correctement 253/256 et certifié 151/256
    Classifié correctement 254/257 et certifié 151/257
    Classifié correctement 255/258 et certifié 151/258
    Classifié correctement 256/259 et certifié 152/259
    Classifié correctement 256/260 et certifié 152/260
    Classifié correctement 257/261 et certifié 153/261
    Classifié correctement 258/262 et certifié 153/262
    Classifié correctement 259/263 et certifié 154/263
    Classifié correctement 260/264 et certifié 155/264
    Classifié correctement 261/265 et certifié 155/265
    Classifié correctement 262/266 et certifié 156/266
    Classifié correctement 263/267 et certifié 156/267
    Classifié correctement 264/268 et certifié 157/268
    Classifié correctement 265/269 et certifié 158/269
    Classifié correctement 266/270 et certifié 159/270
    Classifié correctement 267/271 et certifié 160/271
    Classifié correctement 268/272 et certifié 161/272
    Classifié correctement 269/273 et certifié 162/273
    Classifié correctement 270/274 et certifié 163/274
    Classifié correctement 271/275 et certifié 164/275
    Classifié correctement 272/276 et certifié 165/276
    Classifié correctement 273/277 et certifié 166/277
    Classifié correctement 274/278 et certifié 167/278
    Classifié correctement 275/279 et certifié 168/279
    Classifié correctement 276/280 et certifié 169/280
    Classifié correctement 277/281 et certifié 170/281
    Classifié correctement 278/282 et certifié 171/282
    Classifié correctement 279/283 et certifié 171/283
    Classifié correctement 280/284 et certifié 172/284
    Classifié correctement 281/285 et certifié 173/285
    Classifié correctement 282/286 et certifié 174/286
    Classifié correctement 283/287 et certifié 175/287
    Classifié correctement 284/288 et certifié 175/288
    Classifié correctement 285/289 et certifié 175/289
    Classifié correctement 286/290 et certifié 175/290
    Classifié correctement 286/291 et certifié 175/291
    Classifié correctement 287/292 et certifié 176/292
    Classifié correctement 288/293 et certifié 177/293
    Classifié correctement 289/294 et certifié 178/294
    Classifié correctement 290/295 et certifié 179/295
    Classifié correctement 291/296 et certifié 180/296
    Classifié correctement 292/297 et certifié 181/297
    Classifié correctement 293/298 et certifié 182/298
    Classifié correctement 294/299 et certifié 183/299
    Classifié correctement 295/300 et certifié 183/300
    Classifié correctement 296/301 et certifié 183/301
    Classifié correctement 297/302 et certifié 183/302
    Classifié correctement 298/303 et certifié 184/303
    Classifié correctement 299/304 et certifié 185/304
    Classifié correctement 300/305 et certifié 185/305
    Classifié correctement 301/306 et certifié 186/306
    Classifié correctement 302/307 et certifié 187/307
    Classifié correctement 303/308 et certifié 187/308
    Classifié correctement 304/309 et certifié 188/309
    Classifié correctement 305/310 et certifié 189/310
    Classifié correctement 306/311 et certifié 190/311
    Classifié correctement 307/312 et certifié 191/312
    Classifié correctement 308/313 et certifié 192/313
    Classifié correctement 309/314 et certifié 192/314
    Classifié correctement 310/315 et certifié 193/315
    Classifié correctement 311/316 et certifié 194/316
    Classifié correctement 312/317 et certifié 195/317
    Classifié correctement 313/318 et certifié 195/318
    Classifié correctement 314/319 et certifié 195/319
    Classifié correctement 315/320 et certifié 196/320
    Classifié correctement 315/321 et certifié 196/321
    Classifié correctement 315/322 et certifié 196/322
    Classifié correctement 316/323 et certifié 197/323
    Classifié correctement 317/324 et certifié 198/324
    Classifié correctement 318/325 et certifié 198/325
    Classifié correctement 319/326 et certifié 198/326
    Classifié correctement 320/327 et certifié 198/327
    Classifié correctement 321/328 et certifié 199/328
    Classifié correctement 322/329 et certifié 200/329
    Classifié correctement 323/330 et certifié 201/330
    Classifié correctement 324/331 et certifié 202/331
    Classifié correctement 325/332 et certifié 203/332
    Classifié correctement 326/333 et certifié 204/333
    Classifié correctement 327/334 et certifié 205/334
    Classifié correctement 328/335 et certifié 206/335
    Classifié correctement 329/336 et certifié 206/336
    Classifié correctement 330/337 et certifié 206/337
    Classifié correctement 331/338 et certifié 206/338
    Classifié correctement 332/339 et certifié 207/339
    Classifié correctement 333/340 et certifié 208/340
    Classifié correctement 333/341 et certifié 208/341
    Classifié correctement 334/342 et certifié 208/342
    Classifié correctement 335/343 et certifié 208/343
    Classifié correctement 336/344 et certifié 209/344
    Classifié correctement 337/345 et certifié 209/345
    Classifié correctement 338/346 et certifié 210/346
    Classifié correctement 339/347 et certifié 211/347
    Classifié correctement 340/348 et certifié 212/348
    Classifié correctement 341/349 et certifié 212/349
    Classifié correctement 342/350 et certifié 212/350
    Classifié correctement 343/351 et certifié 212/351
    Classifié correctement 344/352 et certifié 213/352
    Classifié correctement 345/353 et certifié 213/353
    Classifié correctement 346/354 et certifié 214/354
    Classifié correctement 347/355 et certifié 214/355
    Classifié correctement 348/356 et certifié 214/356
    Classifié correctement 349/357 et certifié 215/357
    Classifié correctement 350/358 et certifié 215/358
    Classifié correctement 351/359 et certifié 215/359
    Classifié correctement 351/360 et certifié 215/360
    Classifié correctement 352/361 et certifié 216/361
    Classifié correctement 353/362 et certifié 217/362
    Classifié correctement 354/363 et certifié 217/363
    Classifié correctement 355/364 et certifié 217/364
    Classifié correctement 356/365 et certifié 218/365
    Classifié correctement 357/366 et certifié 219/366
    Classifié correctement 358/367 et certifié 219/367
    Classifié correctement 359/368 et certifié 220/368
    Classifié correctement 360/369 et certifié 221/369
    Classifié correctement 361/370 et certifié 222/370
    Classifié correctement 362/371 et certifié 223/371
    Classifié correctement 363/372 et certifié 224/372
    Classifié correctement 364/373 et certifié 225/373
    Classifié correctement 365/374 et certifié 226/374
    Classifié correctement 366/375 et certifié 227/375
    Classifié correctement 367/376 et certifié 228/376
    Classifié correctement 368/377 et certifié 228/377
    Classifié correctement 369/378 et certifié 228/378
    Classifié correctement 370/379 et certifié 228/379
    Classifié correctement 371/380 et certifié 229/380
    Classifié correctement 372/381 et certifié 230/381
    Classifié correctement 373/382 et certifié 230/382
    Classifié correctement 374/383 et certifié 231/383
    Classifié correctement 375/384 et certifié 231/384
    Classifié correctement 376/385 et certifié 232/385
    Classifié correctement 377/386 et certifié 232/386
    Classifié correctement 378/387 et certifié 232/387
    Classifié correctement 379/388 et certifié 233/388
    Classifié correctement 380/389 et certifié 233/389
    Classifié correctement 381/390 et certifié 233/390
    Classifié correctement 382/391 et certifié 234/391
    Classifié correctement 383/392 et certifié 234/392
    Classifié correctement 384/393 et certifié 235/393
    Classifié correctement 385/394 et certifié 235/394
    Classifié correctement 386/395 et certifié 235/395
    Classifié correctement 387/396 et certifié 236/396
    Classifié correctement 388/397 et certifié 237/397
    Classifié correctement 389/398 et certifié 237/398
    Classifié correctement 390/399 et certifié 238/399
    Classifié correctement 391/400 et certifié 239/400
    Classifié correctement 392/401 et certifié 239/401
    Classifié correctement 393/402 et certifié 239/402
    Classifié correctement 394/403 et certifié 240/403
    Classifié correctement 395/404 et certifié 240/404
    Classifié correctement 396/405 et certifié 240/405
    Classifié correctement 397/406 et certifié 241/406
    Classifié correctement 398/407 et certifié 241/407
    Classifié correctement 399/408 et certifié 242/408
    Classifié correctement 400/409 et certifié 243/409
    Classifié correctement 401/410 et certifié 243/410
    Classifié correctement 402/411 et certifié 244/411
    Classifié correctement 403/412 et certifié 244/412
    Classifié correctement 404/413 et certifié 244/413
    Classifié correctement 405/414 et certifié 245/414
    Classifié correctement 406/415 et certifié 245/415
    Classifié correctement 407/416 et certifié 246/416
    Classifié correctement 408/417 et certifié 247/417
    Classifié correctement 409/418 et certifié 247/418
    Classifié correctement 410/419 et certifié 248/419
    Classifié correctement 411/420 et certifié 248/420
    Classifié correctement 412/421 et certifié 248/421
    Classifié correctement 413/422 et certifié 248/422
    Classifié correctement 414/423 et certifié 249/423
    Classifié correctement 415/424 et certifié 250/424
    Classifié correctement 416/425 et certifié 251/425
    Classifié correctement 417/426 et certifié 252/426
    Classifié correctement 418/427 et certifié 252/427
    Classifié correctement 419/428 et certifié 252/428
    Classifié correctement 420/429 et certifié 253/429
    Classifié correctement 421/430 et certifié 254/430
    Classifié correctement 422/431 et certifié 254/431
    Classifié correctement 423/432 et certifié 254/432
    Classifié correctement 424/433 et certifié 254/433
    Classifié correctement 425/434 et certifié 255/434
    Classifié correctement 426/435 et certifié 255/435
    Classifié correctement 427/436 et certifié 255/436
    Classifié correctement 428/437 et certifié 256/437
    Classifié correctement 429/438 et certifié 257/438
    Classifié correctement 430/439 et certifié 258/439
    Classifié correctement 431/440 et certifié 258/440
    Classifié correctement 432/441 et certifié 259/441
    Classifié correctement 433/442 et certifié 260/442
    Classifié correctement 434/443 et certifié 261/443
    Classifié correctement 435/444 et certifié 261/444
    Classifié correctement 436/445 et certifié 261/445
    Classifié correctement 436/446 et certifié 261/446
    Classifié correctement 437/447 et certifié 262/447
    Classifié correctement 438/448 et certifié 262/448
    Classifié correctement 439/449 et certifié 262/449
    Classifié correctement 439/450 et certifié 262/450
    Classifié correctement 440/451 et certifié 262/451
    Classifié correctement 441/452 et certifié 263/452
    Classifié correctement 442/453 et certifié 263/453
    Classifié correctement 443/454 et certifié 264/454
    Classifié correctement 444/455 et certifié 265/455
    Classifié correctement 445/456 et certifié 266/456
    Classifié correctement 446/457 et certifié 266/457
    Classifié correctement 447/458 et certifié 266/458
    Classifié correctement 448/459 et certifié 267/459
    Classifié correctement 449/460 et certifié 268/460
    Classifié correctement 450/461 et certifié 268/461
    Classifié correctement 451/462 et certifié 269/462
    Classifié correctement 452/463 et certifié 270/463
    Classifié correctement 453/464 et certifié 271/464
    Classifié correctement 454/465 et certifié 272/465
    Classifié correctement 455/466 et certifié 273/466
    Classifié correctement 456/467 et certifié 274/467
    Classifié correctement 457/468 et certifié 275/468
    Classifié correctement 458/469 et certifié 275/469
    Classifié correctement 459/470 et certifié 276/470
    Classifié correctement 460/471 et certifié 277/471
    Classifié correctement 461/472 et certifié 277/472
    Classifié correctement 462/473 et certifié 278/473
    Classifié correctement 463/474 et certifié 278/474
    Classifié correctement 464/475 et certifié 279/475
    Classifié correctement 465/476 et certifié 280/476
    Classifié correctement 466/477 et certifié 280/477
    Classifié correctement 467/478 et certifié 281/478
    Classifié correctement 468/479 et certifié 281/479
    Classifié correctement 469/480 et certifié 281/480
    Classifié correctement 470/481 et certifié 281/481
    Classifié correctement 471/482 et certifié 282/482
    Classifié correctement 472/483 et certifié 283/483
    Classifié correctement 473/484 et certifié 283/484
    Classifié correctement 474/485 et certifié 283/485
    Classifié correctement 475/486 et certifié 284/486
    Classifié correctement 476/487 et certifié 285/487
    Classifié correctement 477/488 et certifié 286/488
    Classifié correctement 478/489 et certifié 286/489
    Classifié correctement 479/490 et certifié 286/490
    Classifié correctement 480/491 et certifié 286/491
    Classifié correctement 481/492 et certifié 287/492
    Classifié correctement 482/493 et certifié 287/493
    Classifié correctement 483/494 et certifié 288/494
    Classifié correctement 484/495 et certifié 289/495
    Classifié correctement 484/496 et certifié 289/496
    Classifié correctement 485/497 et certifié 290/497
    Classifié correctement 486/498 et certifié 290/498
    Classifié correctement 487/499 et certifié 291/499
    Classifié correctement 488/500 et certifié 291/500
    Classifié correctement 489/501 et certifié 292/501
    Classifié correctement 490/502 et certifié 293/502
    Classifié correctement 491/503 et certifié 293/503
    Classifié correctement 492/504 et certifié 294/504
    Classifié correctement 493/505 et certifié 294/505
    Classifié correctement 494/506 et certifié 294/506
    Classifié correctement 495/507 et certifié 294/507
    Classifié correctement 496/508 et certifié 294/508
    Classifié correctement 497/509 et certifié 294/509
    Classifié correctement 498/510 et certifié 295/510
    Classifié correctement 499/511 et certifié 295/511
    Classifié correctement 500/512 et certifié 295/512
    Classifié correctement 501/513 et certifié 296/513
    Classifié correctement 502/514 et certifié 297/514
    Classifié correctement 503/515 et certifié 298/515
    Classifié correctement 504/516 et certifié 298/516
    Classifié correctement 505/517 et certifié 299/517
    Classifié correctement 506/518 et certifié 300/518
    Classifié correctement 507/519 et certifié 301/519
    Classifié correctement 508/520 et certifié 301/520
    Classifié correctement 509/521 et certifié 301/521
    Classifié correctement 510/522 et certifié 301/522
    Classifié correctement 511/523 et certifié 301/523
    Classifié correctement 512/524 et certifié 301/524
    Classifié correctement 513/525 et certifié 301/525
    Classifié correctement 514/526 et certifié 302/526
    Classifié correctement 515/527 et certifié 303/527
    Classifié correctement 516/528 et certifié 304/528
    Classifié correctement 517/529 et certifié 304/529
    Classifié correctement 518/530 et certifié 305/530
    Classifié correctement 519/531 et certifié 305/531
    Classifié correctement 520/532 et certifié 305/532
    Classifié correctement 521/533 et certifié 305/533
    Classifié correctement 522/534 et certifié 306/534
    Classifié correctement 523/535 et certifié 307/535
    Classifié correctement 524/536 et certifié 308/536
    Classifié correctement 525/537 et certifié 308/537
    Classifié correctement 526/538 et certifié 308/538
    Classifié correctement 527/539 et certifié 308/539
    Classifié correctement 528/540 et certifié 309/540
    Classifié correctement 529/541 et certifié 310/541
    Classifié correctement 530/542 et certifié 311/542
    Classifié correctement 531/543 et certifié 311/543
    Classifié correctement 532/544 et certifié 311/544
    Classifié correctement 533/545 et certifié 312/545
    Classifié correctement 534/546 et certifié 313/546
    Classifié correctement 535/547 et certifié 314/547
    Classifié correctement 536/548 et certifié 314/548
    Classifié correctement 537/549 et certifié 315/549
    Classifié correctement 538/550 et certifié 316/550
    Classifié correctement 539/551 et certifié 316/551
    Classifié correctement 539/552 et certifié 316/552
    Classifié correctement 540/553 et certifié 316/553
    Classifié correctement 541/554 et certifié 316/554
    Classifié correctement 542/555 et certifié 317/555
    Classifié correctement 543/556 et certifié 318/556
    Classifié correctement 544/557 et certifié 319/557
    Classifié correctement 545/558 et certifié 320/558
    Classifié correctement 546/559 et certifié 320/559
    Classifié correctement 547/560 et certifié 321/560
    Classifié correctement 548/561 et certifié 321/561
    Classifié correctement 549/562 et certifié 322/562
    Classifié correctement 550/563 et certifié 322/563
    Classifié correctement 551/564 et certifié 322/564
    Classifié correctement 552/565 et certifié 323/565
    Classifié correctement 553/566 et certifié 323/566
    Classifié correctement 554/567 et certifié 323/567
    Classifié correctement 555/568 et certifié 324/568
    Classifié correctement 556/569 et certifié 325/569
    Classifié correctement 557/570 et certifié 325/570
    Classifié correctement 558/571 et certifié 326/571
    Classifié correctement 559/572 et certifié 326/572
    Classifié correctement 560/573 et certifié 326/573
    Classifié correctement 561/574 et certifié 327/574
    Classifié correctement 562/575 et certifié 328/575
    Classifié correctement 563/576 et certifié 328/576
    Classifié correctement 564/577 et certifié 329/577
    Classifié correctement 565/578 et certifié 330/578
    Classifié correctement 566/579 et certifié 330/579
    Classifié correctement 567/580 et certifié 330/580
    Classifié correctement 568/581 et certifié 331/581
    Classifié correctement 569/582 et certifié 332/582
    Classifié correctement 569/583 et certifié 332/583
    Classifié correctement 570/584 et certifié 332/584
    Classifié correctement 571/585 et certifié 333/585
    Classifié correctement 572/586 et certifié 334/586
    Classifié correctement 573/587 et certifié 335/587
    Classifié correctement 574/588 et certifié 336/588
    Classifié correctement 575/589 et certifié 336/589
    Classifié correctement 576/590 et certifié 336/590
    Classifié correctement 577/591 et certifié 337/591
    Classifié correctement 577/592 et certifié 337/592
    Classifié correctement 578/593 et certifié 338/593
    Classifié correctement 579/594 et certifié 338/594
    Classifié correctement 580/595 et certifié 339/595
    Classifié correctement 581/596 et certifié 340/596
    Classifié correctement 582/597 et certifié 341/597
    Classifié correctement 583/598 et certifié 341/598
    Classifié correctement 584/599 et certifié 341/599
    Classifié correctement 585/600 et certifié 342/600
    Classifié correctement 586/601 et certifié 343/601
    Classifié correctement 587/602 et certifié 343/602
    Classifié correctement 588/603 et certifié 344/603
    Classifié correctement 589/604 et certifié 344/604
    Classifié correctement 590/605 et certifié 345/605
    Classifié correctement 591/606 et certifié 345/606
    Classifié correctement 592/607 et certifié 345/607
    Classifié correctement 593/608 et certifié 346/608
    Classifié correctement 594/609 et certifié 347/609
    Classifié correctement 595/610 et certifié 348/610
    Classifié correctement 596/611 et certifié 348/611
    Classifié correctement 597/612 et certifié 349/612
    Classifié correctement 598/613 et certifié 350/613
    Classifié correctement 599/614 et certifié 350/614
    Classifié correctement 600/615 et certifié 350/615
    Classifié correctement 601/616 et certifié 351/616
    Classifié correctement 602/617 et certifié 351/617
    Classifié correctement 603/618 et certifié 351/618
    Classifié correctement 604/619 et certifié 352/619
    Classifié correctement 605/620 et certifié 352/620
    Classifié correctement 606/621 et certifié 353/621
    Classifié correctement 607/622 et certifié 354/622
    Classifié correctement 608/623 et certifié 355/623
    Classifié correctement 609/624 et certifié 356/624
    Classifié correctement 610/625 et certifié 356/625
    Classifié correctement 611/626 et certifié 356/626
    Classifié correctement 612/627 et certifié 357/627
    Classifié correctement 613/628 et certifié 357/628
    Classifié correctement 614/629 et certifié 357/629
    Classifié correctement 615/630 et certifié 357/630
    Classifié correctement 616/631 et certifié 357/631
    Classifié correctement 617/632 et certifié 358/632
    Classifié correctement 618/633 et certifié 359/633
    Classifié correctement 619/634 et certifié 360/634
    Classifié correctement 620/635 et certifié 361/635
    Classifié correctement 621/636 et certifié 361/636
    Classifié correctement 622/637 et certifié 362/637
    Classifié correctement 623/638 et certifié 363/638
    Classifié correctement 624/639 et certifié 363/639
    Classifié correctement 625/640 et certifié 363/640
    Classifié correctement 626/641 et certifié 363/641
    Classifié correctement 627/642 et certifié 364/642
    Classifié correctement 628/643 et certifié 365/643
    Classifié correctement 629/644 et certifié 366/644
    Classifié correctement 630/645 et certifié 367/645
    Classifié correctement 631/646 et certifié 367/646
    Classifié correctement 632/647 et certifié 367/647
    Classifié correctement 633/648 et certifié 367/648
    Classifié correctement 634/649 et certifié 368/649
    Classifié correctement 635/650 et certifié 368/650
    Classifié correctement 636/651 et certifié 369/651
    Classifié correctement 637/652 et certifié 370/652
    Classifié correctement 638/653 et certifié 371/653
    Classifié correctement 639/654 et certifié 372/654
    Classifié correctement 640/655 et certifié 373/655
    Classifié correctement 641/656 et certifié 374/656
    Classifié correctement 642/657 et certifié 375/657
    Classifié correctement 643/658 et certifié 376/658
    Classifié correctement 644/659 et certifié 376/659
    Classifié correctement 644/660 et certifié 376/660
    Classifié correctement 645/661 et certifié 377/661
    Classifié correctement 646/662 et certifié 378/662
    Classifié correctement 647/663 et certifié 379/663
    Classifié correctement 648/664 et certifié 380/664
    Classifié correctement 649/665 et certifié 381/665
    Classifié correctement 650/666 et certifié 382/666
    Classifié correctement 651/667 et certifié 382/667
    Classifié correctement 652/668 et certifié 382/668
    Classifié correctement 653/669 et certifié 383/669
    Classifié correctement 654/670 et certifié 384/670
    Classifié correctement 655/671 et certifié 385/671
    Classifié correctement 656/672 et certifié 386/672
    Classifié correctement 657/673 et certifié 386/673
    Classifié correctement 658/674 et certifié 386/674
    Classifié correctement 659/675 et certifié 386/675
    Classifié correctement 660/676 et certifié 386/676
    Classifié correctement 661/677 et certifié 387/677
    Classifié correctement 662/678 et certifié 388/678
    Classifié correctement 663/679 et certifié 388/679
    Classifié correctement 664/680 et certifié 389/680
    Classifié correctement 665/681 et certifié 390/681
    Classifié correctement 666/682 et certifié 391/682
    Classifié correctement 667/683 et certifié 392/683
    Classifié correctement 668/684 et certifié 392/684
    Classifié correctement 668/685 et certifié 392/685
    Classifié correctement 669/686 et certifié 392/686
    Classifié correctement 670/687 et certifié 393/687
    Classifié correctement 671/688 et certifié 394/688
    Classifié correctement 672/689 et certifié 395/689
    Classifié correctement 673/690 et certifié 395/690
    Classifié correctement 674/691 et certifié 396/691
    Classifié correctement 674/692 et certifié 396/692
    Classifié correctement 675/693 et certifié 396/693
    Classifié correctement 676/694 et certifié 397/694
    Classifié correctement 677/695 et certifié 397/695
    Classifié correctement 678/696 et certifié 397/696
    Classifié correctement 679/697 et certifié 397/697
    Classifié correctement 680/698 et certifié 398/698
    Classifié correctement 681/699 et certifié 398/699
    Classifié correctement 682/700 et certifié 398/700
    Classifié correctement 683/701 et certifié 398/701
    Classifié correctement 684/702 et certifié 399/702
    Classifié correctement 685/703 et certifié 400/703
    Classifié correctement 686/704 et certifié 400/704
    Classifié correctement 687/705 et certifié 401/705
    Classifié correctement 688/706 et certifié 402/706
    Classifié correctement 689/707 et certifié 403/707
    Classifié correctement 689/708 et certifié 403/708
    Classifié correctement 690/709 et certifié 404/709
    Classifié correctement 691/710 et certifié 405/710
    Classifié correctement 692/711 et certifié 406/711
    Classifié correctement 693/712 et certifié 407/712
    Classifié correctement 694/713 et certifié 407/713
    Classifié correctement 695/714 et certifié 408/714
    Classifié correctement 696/715 et certifié 408/715
    Classifié correctement 697/716 et certifié 409/716
    Classifié correctement 698/717 et certifié 409/717
    Classifié correctement 698/718 et certifié 409/718
    Classifié correctement 699/719 et certifié 410/719
    Classifié correctement 700/720 et certifié 411/720
    Classifié correctement 700/721 et certifié 411/721
    Classifié correctement 701/722 et certifié 412/722
    Classifié correctement 702/723 et certifié 413/723
    Classifié correctement 703/724 et certifié 413/724
    Classifié correctement 704/725 et certifié 414/725
    Classifié correctement 705/726 et certifié 414/726
    Classifié correctement 706/727 et certifié 414/727
    Classifié correctement 707/728 et certifié 415/728
    Classifié correctement 708/729 et certifié 416/729
    Classifié correctement 709/730 et certifié 417/730
    Classifié correctement 710/731 et certifié 418/731
    Classifié correctement 711/732 et certifié 419/732
    Classifié correctement 712/733 et certifié 420/733
    Classifié correctement 713/734 et certifié 421/734
    Classifié correctement 714/735 et certifié 422/735
    Classifié correctement 715/736 et certifié 423/736
    Classifié correctement 716/737 et certifié 424/737
    Classifié correctement 717/738 et certifié 425/738
    Classifié correctement 718/739 et certifié 425/739
    Classifié correctement 719/740 et certifié 426/740
    Classifié correctement 719/741 et certifié 426/741
    Classifié correctement 720/742 et certifié 427/742
    Classifié correctement 721/743 et certifié 428/743
    Classifié correctement 722/744 et certifié 429/744
    Classifié correctement 723/745 et certifié 430/745
    Classifié correctement 724/746 et certifié 430/746
    Classifié correctement 725/747 et certifié 430/747
    Classifié correctement 726/748 et certifié 431/748
    Classifié correctement 727/749 et certifié 431/749
    Classifié correctement 728/750 et certifié 431/750
    Classifié correctement 729/751 et certifié 432/751
    Classifié correctement 730/752 et certifié 433/752
    Classifié correctement 731/753 et certifié 434/753
    Classifié correctement 732/754 et certifié 435/754
    Classifié correctement 733/755 et certifié 435/755
    Classifié correctement 734/756 et certifié 435/756
    Classifié correctement 735/757 et certifié 436/757
    Classifié correctement 736/758 et certifié 437/758
    Classifié correctement 737/759 et certifié 438/759
    Classifié correctement 738/760 et certifié 438/760
    Classifié correctement 739/761 et certifié 438/761
    Classifié correctement 740/762 et certifié 438/762
    Classifié correctement 741/763 et certifié 439/763
    Classifié correctement 742/764 et certifié 440/764
    Classifié correctement 743/765 et certifié 441/765
    Classifié correctement 744/766 et certifié 442/766
    Classifié correctement 745/767 et certifié 443/767
    Classifié correctement 746/768 et certifié 443/768
    Classifié correctement 747/769 et certifié 444/769
    Classifié correctement 748/770 et certifié 445/770
    Classifié correctement 749/771 et certifié 446/771
    Classifié correctement 750/772 et certifié 446/772
    Classifié correctement 751/773 et certifié 447/773
    Classifié correctement 752/774 et certifié 448/774
    Classifié correctement 753/775 et certifié 448/775
    Classifié correctement 754/776 et certifié 449/776
    Classifié correctement 755/777 et certifié 450/777
    Classifié correctement 756/778 et certifié 450/778
    Classifié correctement 757/779 et certifié 451/779
    Classifié correctement 758/780 et certifié 452/780
    Classifié correctement 759/781 et certifié 453/781
    Classifié correctement 760/782 et certifié 453/782
    Classifié correctement 761/783 et certifié 454/783
    Classifié correctement 762/784 et certifié 454/784
    Classifié correctement 763/785 et certifié 455/785
    Classifié correctement 764/786 et certifié 455/786
    Classifié correctement 765/787 et certifié 455/787
    Classifié correctement 766/788 et certifié 456/788
    Classifié correctement 767/789 et certifié 457/789
    Classifié correctement 768/790 et certifié 457/790
    Classifié correctement 769/791 et certifié 457/791
    Classifié correctement 770/792 et certifié 457/792
    Classifié correctement 771/793 et certifié 458/793
    Classifié correctement 772/794 et certifié 459/794
    Classifié correctement 773/795 et certifié 460/795
    Classifié correctement 774/796 et certifié 460/796
    Classifié correctement 775/797 et certifié 461/797
    Classifié correctement 776/798 et certifié 461/798
    Classifié correctement 777/799 et certifié 461/799
    Classifié correctement 778/800 et certifié 462/800
    Classifié correctement 779/801 et certifié 462/801
    Classifié correctement 780/802 et certifié 463/802
    Classifié correctement 781/803 et certifié 464/803
    Classifié correctement 782/804 et certifié 465/804
    Classifié correctement 783/805 et certifié 466/805
    Classifié correctement 784/806 et certifié 467/806
    Classifié correctement 785/807 et certifié 468/807
    Classifié correctement 786/808 et certifié 469/808
    Classifié correctement 787/809 et certifié 470/809
    Classifié correctement 788/810 et certifié 470/810
    Classifié correctement 788/811 et certifié 470/811
    Classifié correctement 789/812 et certifié 471/812
    Classifié correctement 790/813 et certifié 472/813
    Classifié correctement 791/814 et certifié 472/814
    Classifié correctement 792/815 et certifié 473/815
    Classifié correctement 793/816 et certifié 474/816
    Classifié correctement 794/817 et certifié 475/817
    Classifié correctement 795/818 et certifié 476/818
    Classifié correctement 796/819 et certifié 477/819
    Classifié correctement 797/820 et certifié 478/820
    Classifié correctement 798/821 et certifié 479/821
    Classifié correctement 799/822 et certifié 480/822
    Classifié correctement 800/823 et certifié 481/823
    Classifié correctement 801/824 et certifié 482/824
    Classifié correctement 802/825 et certifié 482/825
    Classifié correctement 803/826 et certifié 483/826
    Classifié correctement 804/827 et certifié 483/827
    Classifié correctement 805/828 et certifié 483/828
    Classifié correctement 806/829 et certifié 484/829
    Classifié correctement 807/830 et certifié 484/830
    Classifié correctement 808/831 et certifié 484/831
    Classifié correctement 809/832 et certifié 484/832
    Classifié correctement 810/833 et certifié 484/833
    Classifié correctement 811/834 et certifié 484/834
    Classifié correctement 812/835 et certifié 484/835
    Classifié correctement 813/836 et certifié 484/836
    Classifié correctement 814/837 et certifié 484/837
    Classifié correctement 815/838 et certifié 484/838
    Classifié correctement 816/839 et certifié 485/839
    Classifié correctement 816/840 et certifié 485/840
    Classifié correctement 817/841 et certifié 485/841
    Classifié correctement 818/842 et certifié 485/842
    Classifié correctement 819/843 et certifié 485/843
    Classifié correctement 820/844 et certifié 486/844
    Classifié correctement 820/845 et certifié 486/845
    Classifié correctement 821/846 et certifié 487/846
    Classifié correctement 822/847 et certifié 487/847
    Classifié correctement 823/848 et certifié 488/848
    Classifié correctement 824/849 et certifié 488/849
    Classifié correctement 825/850 et certifié 489/850
    Classifié correctement 826/851 et certifié 489/851
    Classifié correctement 827/852 et certifié 490/852
    Classifié correctement 828/853 et certifié 490/853
    Classifié correctement 829/854 et certifié 491/854
    Classifié correctement 830/855 et certifié 492/855
    Classifié correctement 831/856 et certifié 493/856
    Classifié correctement 832/857 et certifié 494/857
    Classifié correctement 833/858 et certifié 494/858
    Classifié correctement 834/859 et certifié 495/859
    Classifié correctement 835/860 et certifié 495/860
    Classifié correctement 836/861 et certifié 496/861
    Classifié correctement 837/862 et certifié 497/862
    Classifié correctement 838/863 et certifié 497/863
    Classifié correctement 839/864 et certifié 497/864
    Classifié correctement 840/865 et certifié 498/865
    Classifié correctement 841/866 et certifié 499/866
    Classifié correctement 842/867 et certifié 499/867
    Classifié correctement 843/868 et certifié 500/868
    Classifié correctement 844/869 et certifié 501/869
    Classifié correctement 845/870 et certifié 502/870
    Classifié correctement 846/871 et certifié 502/871
    Classifié correctement 847/872 et certifié 503/872
    Classifié correctement 848/873 et certifié 503/873
    Classifié correctement 849/874 et certifié 504/874
    Classifié correctement 850/875 et certifié 504/875
    Classifié correctement 851/876 et certifié 505/876
    Classifié correctement 852/877 et certifié 506/877
    Classifié correctement 853/878 et certifié 507/878
    Classifié correctement 854/879 et certifié 507/879
    Classifié correctement 855/880 et certifié 507/880
    Classifié correctement 856/881 et certifié 507/881
    Classifié correctement 857/882 et certifié 507/882
    Classifié correctement 858/883 et certifié 507/883
    Classifié correctement 859/884 et certifié 507/884
    Classifié correctement 860/885 et certifié 508/885
    Classifié correctement 861/886 et certifié 508/886
    Classifié correctement 862/887 et certifié 509/887
    Classifié correctement 863/888 et certifié 510/888
    Classifié correctement 864/889 et certifié 511/889
    Classifié correctement 865/890 et certifié 511/890
    Classifié correctement 866/891 et certifié 511/891
    Classifié correctement 867/892 et certifié 512/892
    Classifié correctement 868/893 et certifié 513/893
    Classifié correctement 869/894 et certifié 513/894
    Classifié correctement 870/895 et certifié 513/895
    Classifié correctement 871/896 et certifié 514/896
    Classifié correctement 872/897 et certifié 514/897
    Classifié correctement 873/898 et certifié 515/898
    Classifié correctement 874/899 et certifié 515/899
    Classifié correctement 875/900 et certifié 516/900
    Classifié correctement 876/901 et certifié 516/901
    Classifié correctement 877/902 et certifié 516/902
    Classifié correctement 878/903 et certifié 516/903
    Classifié correctement 879/904 et certifié 517/904
    Classifié correctement 880/905 et certifié 518/905
    Classifié correctement 881/906 et certifié 519/906
    Classifié correctement 882/907 et certifié 519/907
    Classifié correctement 883/908 et certifié 519/908
    Classifié correctement 884/909 et certifié 520/909
    Classifié correctement 885/910 et certifié 520/910
    Classifié correctement 886/911 et certifié 521/911
    Classifié correctement 887/912 et certifié 522/912
    Classifié correctement 888/913 et certifié 523/913
    Classifié correctement 889/914 et certifié 524/914
    Classifié correctement 890/915 et certifié 525/915
    Classifié correctement 891/916 et certifié 525/916
    Classifié correctement 892/917 et certifié 526/917
    Classifié correctement 893/918 et certifié 527/918
    Classifié correctement 894/919 et certifié 527/919
    Classifié correctement 895/920 et certifié 528/920
    Classifié correctement 896/921 et certifié 528/921
    Classifié correctement 897/922 et certifié 529/922
    Classifié correctement 898/923 et certifié 529/923
    Classifié correctement 899/924 et certifié 530/924
    Classifié correctement 899/925 et certifié 530/925
    Classifié correctement 900/926 et certifié 531/926
    Classifié correctement 900/927 et certifié 531/927
    Classifié correctement 901/928 et certifié 532/928
    Classifié correctement 902/929 et certifié 533/929
    Classifié correctement 903/930 et certifié 533/930
    Classifié correctement 904/931 et certifié 533/931
    Classifié correctement 905/932 et certifié 533/932
    Classifié correctement 906/933 et certifié 534/933
    Classifié correctement 907/934 et certifié 535/934
    Classifié correctement 908/935 et certifié 536/935
    Classifié correctement 909/936 et certifié 537/936
    Classifié correctement 910/937 et certifié 537/937
    Classifié correctement 911/938 et certifié 538/938
    Classifié correctement 912/939 et certifié 538/939
    Classifié correctement 913/940 et certifié 538/940
    Classifié correctement 914/941 et certifié 539/941
    Classifié correctement 915/942 et certifié 540/942
    Classifié correctement 916/943 et certifié 541/943
    Classifié correctement 917/944 et certifié 542/944
    Classifié correctement 918/945 et certifié 542/945
    Classifié correctement 919/946 et certifié 543/946
    Classifié correctement 920/947 et certifié 543/947
    Classifié correctement 920/948 et certifié 543/948
    Classifié correctement 921/949 et certifié 544/949
    Classifié correctement 922/950 et certifié 545/950
    Classifié correctement 922/951 et certifié 545/951
    Classifié correctement 923/952 et certifié 545/952
    Classifié correctement 924/953 et certifié 545/953
    Classifié correctement 925/954 et certifié 546/954
    Classifié correctement 926/955 et certifié 546/955
    Classifié correctement 927/956 et certifié 546/956
    Classifié correctement 928/957 et certifié 546/957
    Classifié correctement 929/958 et certifié 547/958
    Classifié correctement 930/959 et certifié 547/959
    Classifié correctement 931/960 et certifié 547/960
    Classifié correctement 932/961 et certifié 547/961
    Classifié correctement 933/962 et certifié 548/962
    Classifié correctement 934/963 et certifié 548/963
    Classifié correctement 935/964 et certifié 548/964
    Classifié correctement 936/965 et certifié 548/965
    Classifié correctement 936/966 et certifié 548/966
    Classifié correctement 937/967 et certifié 548/967
    Classifié correctement 938/968 et certifié 549/968
    Classifié correctement 939/969 et certifié 550/969
    Classifié correctement 940/970 et certifié 550/970
    Classifié correctement 941/971 et certifié 551/971
    Classifié correctement 942/972 et certifié 552/972
    Classifié correctement 943/973 et certifié 553/973
    Classifié correctement 944/974 et certifié 554/974
    Classifié correctement 945/975 et certifié 555/975
    Classifié correctement 946/976 et certifié 555/976
    Classifié correctement 947/977 et certifié 555/977
    Classifié correctement 948/978 et certifié 556/978
    Classifié correctement 949/979 et certifié 557/979
    Classifié correctement 950/980 et certifié 557/980
    Classifié correctement 951/981 et certifié 558/981
    Classifié correctement 952/982 et certifié 559/982
    Classifié correctement 952/983 et certifié 559/983
    Classifié correctement 953/984 et certifié 560/984
    Classifié correctement 954/985 et certifié 560/985
    Classifié correctement 955/986 et certifié 561/986
    Classifié correctement 956/987 et certifié 562/987
    Classifié correctement 957/988 et certifié 563/988
    Classifié correctement 958/989 et certifié 563/989
    Classifié correctement 959/990 et certifié 564/990
    Classifié correctement 960/991 et certifié 564/991
    Classifié correctement 961/992 et certifié 565/992
    Classifié correctement 962/993 et certifié 565/993
    Classifié correctement 963/994 et certifié 566/994
    Classifié correctement 964/995 et certifié 566/995
    Classifié correctement 965/996 et certifié 567/996
    Classifié correctement 966/997 et certifié 568/997
    Classifié correctement 967/998 et certifié 569/998
    Classifié correctement 968/999 et certifié 570/999
    Classifié correctement 969/1000 et certifié 570/1000



```python
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=criterion,
    optimizer=opt,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

attack = ProjectedGradientDescent(classifier, eps=0.05, eps_step=0.01, verbose=False)
x_train_adv = attack.generate(x_test[:sample_size].astype('float32'))
y_adv_pred = classifier.predict(torch.from_numpy(x_train_adv).float().to(device))
y_adv_pred = np.argmax(y_adv_pred, axis=1)
print('Test acc: ', np.mean(y_adv_pred == y_test[:sample_size]) * 100)
```

    Test acc:  83.5


On voit clairement que l'accuracy empirique est bien plus élévée que l'accuracy certifiée, ce qui tombe sous le sens.

Cependant, comme évoqué en introduction, les méthodes incomplètes ont le désavantage de générer des faux négatif ( certifié non robuste alors que c'est le cas), ce qui a tendance à sous-estimé la veritable performance "certifiée robuste".
