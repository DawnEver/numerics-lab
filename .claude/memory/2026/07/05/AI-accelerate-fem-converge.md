> From: https://gemini.google.com/app/89b87e03b504e6bb

# you asked

message time: 2026-07-05 10:41:27

(user, translated) Interpret this: https://monumo.ai/latest/ai-accelerated-fem-solvers-improving-convergence-with-better-initialisation

---

# gemini response

This article comes from the Cambridge deep-tech company **Monumo** (focused on using AI to optimize electric motor design). The core technical point of the article is **how to use AI (deep learning / machine learning) to accelerate traditional finite element analysis (FEM/FEA) solvers, in particular by "providing better initial values (Better Initialisation)" to fundamentally improve the convergence speed of nonlinear solves (Convergence)**.

If you work on electric motor design, electromagnetic field simulation or computational mechanics, you will find this a very hardcore technical breakthrough that hits the pain point directly.

Below is an interpretation of the article's core content:

---

## 1. The core pain point: why is traditional FEM solving so slow?

In electric motor design (e.g. multiphysics simulation, nonlinear electromagnetic field computation), engineers have to adjust shapes and parameters over and over.
* **Nonlinear problems hold back speed**: materials in a motor such as silicon steel laminations have strong **magnetic saturation characteristics (Non-linear Magnetic Saturation)**. This means that when solving Maxwell's equations you cannot get there in one step; you must use iterative algorithms such as the **Newton-Raphson method (Newton-Raphson)**.
* **A "blind" initial guess**: when a traditional FEM solver starts computing a new design, it usually uses an all-zero or very coarse generic state as the **initial guess (Initial Guess)**.
* **Convergence difficulty**: if the initial guess is far from the true physical result, the iterative solver has to take many detours, needing tens or even hundreds of iterations to converge (Convergence). If the design is extreme, it can even **fail to converge (never finish)**. This becomes a huge bottleneck for a company like Monumo that needs to "screen millions of design candidates".

---

## 2. Monumo's solution: an AI-powered "godlike oracle"

Monumo's idea is very clever: **it does not use AI to replace FEM entirely (because a full replacement can hardly guarantee 100% physical accuracy), but lets AI act as FEM's "assistant oracle"**.

### The change in workflow
* **Traditional flow**: input a new motor design $\rightarrow$ give a default all-zero initial value $\rightarrow$ long nonlinear iteration $\rightarrow$ converged output.
* **AI-accelerated flow**: input a new motor design $\rightarrow$ **AI (a neural network) instantly predicts an electromagnetic field distribution very close to reality (as the initial value)** $\rightarrow$ the traditional FEM solver takes over $\rightarrow$ converges rapidly after very few iterations and outputs.

> **Underlying logic**: AI is responsible for fast qualitative work (giving a 90%-correct outline), and traditional high-accuracy FEM is responsible for the final quantitative gap-filling (supplying the last 10% of physical accuracy). This keeps FEM's strict physical accuracy while squeezing out AI's tremendous speed.

---

## 3. Technical breakthroughs and advantages

According to information disclosed by Monumo, this "AI acceleration based on better initial values" brings the following qualitative leaps:

* **Greatly improved convergence speed (Speedup)**: because the initial value is already extremely close to the final answer, the number of Newton iterations drops off a cliff, and simulation time can usually be cut several-fold or even by an order of magnitude.
* **High robustness (fewer errors)**: a good initial value effectively prevents the nonlinear iteration from getting stuck in a local minimum or diverging outright, greatly reducing simulations that abort with errors.
* **Enabling ultra-large-scale design search (Anser® Engine)**: Monumo can achieve the industry legend of "evaluating 3 million motor design candidates in 3 days" precisely because every underlying FEM node is accelerated to the extreme by this AI solver.

---

## 4. Industry implications

Traditional AI + scientific computing (AI4S) often falls into an extreme: trying to replace physical simulation entirely end-to-end with neural networks, but often, because of "hallucinations" or lack of generalization, engineers do not dare adopt it in real production.

This Monumo article shows the **most pragmatic and most easily deployed** physics-AI combination paradigm in industry today: **the neural network provides smart, high-value prior knowledge (initialization), and the deterministic algorithm (FEM) is responsible for strict physical conservation and guarding convergence.**

---

<FollowUp label="Want to learn more about how the neural network is trained to predict such high-accuracy electromagnetic field initial values?" query="How is Monumo's AI model trained across different motor geometries and material properties so that it can accurately predict FEM electromagnetic field initial values?"/>

---

# you asked

message time: 2026-07-05 10:43:10

(user, translated) How is Monumo's AI model trained across different motor geometries and material properties so that it can accurately predict FEM electromagnetic field initial values?

---

# gemini response

For AI to accurately predict electromagnetic field initial values under different geometries and material properties, the core is **turning the continuous variation of the physical world into generic feature inputs that AI can understand**.

Monumo's **Anser® engine** builds, at its foundation, a training framework that deeply fuses "machine learning (ML)" with "traditional finite element analysis (FEM)".

---

## 1. Parameterization and data representation (Data Representation)
AI cannot directly read complex 3D motor CAD drawings; they must be digitized:

* **Free-form geometric parameterization (Free-form Parameterization)**: Monumo abandons traditional fixed templates (such as standard tooth and slot shapes) and instead uses a parametric geometric description. The shapes of the stator (Stator) and rotor (Rotor) are broken down into a parameter matrix made of control points, arcs and thicknesses. This way, any oddly shaped geometry can become a string of continuous numeric inputs.
* **Mapping material properties (Material Mapping)**: the magnetic nonlinearity of nonlinear materials (such as different grades of silicon steel) shows up mainly in the **$B-H$ curve (the relation between magnetic flux density and magnetic field strength)**. The model binds the mathematical representation of these nonlinear curves, the material's saturation point and its permeability as feature vectors (Feature Vectors) to the corresponding geometric mesh, and feeds them to the AI together.

---

## 2. A tremendous "synthetic data" factory: tens of millions of simulations per day
AI accuracy depends on massive data. Monumo's way of solving the "data famine" is very direct -- **generate high-quality data itself**.

* **Parallelized automatic generation**: Monumo's Anser® engine has very strong concurrent computing capacity and set the industry record of **running 10 million simulations per day**.
* **Large-scale space sampling**: they actively sample a high-dimensional space containing millions of "geometric parameters $\times$ material combinations $\times$ excitation currents", and use traditional high-accuracy FEM to solve the final true electromagnetic field for these combinations (such as the $A$ vector potential or $B$ flux density distribution).
* **Aligned labels (Labels)**: these FEM-computed true electromagnetic field mesh data are the perfect training labels.

---

## 3. Core model architecture: surrogate models and graph neural networks
In actual training, Monumo mainly uses two AI model techniques to cope with varying shapes:

### Physics-informed surrogate models (Physics-Informed Surrogate Models)
These models do not guess blindly; they add physical constraints (such as the residual of Maxwell's equations and boundary conditions) into the loss function (Loss Function). This means that even when encountering a geometry never seen before, the initial field predicted by the AI naturally satisfies basic physical laws such as "magnetic flux continuity", without absurd "hallucinations".

### Graph neural networks (GNNs) or geometric deep learning
An ordinary convolutional neural network (CNN) can only handle square, regular images, while a motor has arcs and complex meshes.
* **Mesh as a graph (Mesh as a Graph)**: a GNN treats the motor's FEM mesh as a "graph (Graph)": mesh nodes are nodes, and the physical connections between nodes are edges.
* **Topology adaptivity**: when the motor geometry changes or the mesh deforms or is refined, the GNN can adaptively perform "feature passing" (Message Passing) on these unstructured meshes, thereby precisely capturing the strong influence of abrupt geometric features (such as tooth tips and the air gap) on the electromagnetic field.

---

## 4. Closed-loop active learning (Active Learning)
To keep the AI from failing on unusual, odd motor designs, Monumo introduces an active learning mechanism:

```
[ AI predicts initial value ] ──> [ FEM iteration check ]
       ▲                     │
       │ (slow convergence / large error found)   ▼
[ extract the failing case ] <── [ trigger high-accuracy simulation for more data ]
```

When the Anser® engine is screening millions of candidates, if it finds that some odd shape makes the AI-predicted initial value perform poorly (causing more FEM iterations), the system immediately marks that region as a "high-uncertainty region", automatically calls traditional high-accuracy FEM to compute the correct answer, and then adds this new data back to the training set to "feed" the AI.

---

> **Summary**
> Monumo has in effect built a **self-circulating data super-factory**: it uses massive (tens of millions) multiphysics simulation data to feed a **graph neural network / surrogate model** that can adapt to arbitrary geometry and nonlinear materials. Having been tempered by millions of extreme designs, the AI can naturally see at a glance roughly what the electromagnetic field of a new design looks like, and then give a perfect "godlike initial value".

> Amended 2026-11-17: translated to English (D26) -- approved: "(user, translated) .claude/memory/ and .claude/tasks/ should also be English only"
