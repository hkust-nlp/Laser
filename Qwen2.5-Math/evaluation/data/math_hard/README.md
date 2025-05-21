---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
source_datasets:
- original
task_categories:
- text2text-generation
task_ids: []
pretty_name: Mathematics Aptitude Test of Heuristics (MATH)
tags:
- explanation-generation
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*
  - split: test
    path: test/*
- config_name: algebra
  data_files:
  - split: train
    path: train/algebra.jsonl
  - split: test
    path: test/algebra.jsonl
- config_name: counting_and_probability
  data_files:
  - split: train
    path: train/counting_and_probability.jsonl
  - split: test
    path: test/counting_and_probability.jsonl
- config_name: geometry
  data_files:
  - split: train
    path: train/geometry.jsonl
  - split: test
    path: test/geometry.jsonl
- config_name: intermediate_algebra
  data_files:
  - split: train
    path: train/intermediate_algebra.jsonl
  - split: test
    path: test/intermediate_algebra.jsonl
- config_name: number_theory
  data_files:
  - split: train
    path: train/number_theory.jsonl
  - split: test
    path: test/number_theory.jsonl
- config_name: prealgebra
  data_files:
  - split: train
    path: train/prealgebra.jsonl
  - split: test
    path: test/prealgebra.jsonl
- config_name: precalculus
  data_files:
  - split: train
    path: train/precalculus.jsonl
  - split: test
    path: test/precalculus.jsonl
---

# Dataset Card for Mathematics Aptitude Test of Heuristics, hard subset (MATH-Hard) dataset
## Dataset Description

- **Homepage:** https://github.com/hendrycks/math
- **Repository:** https://github.com/hendrycks/math
- **Paper:** https://arxiv.org/pdf/2103.03874.pdf
- **Leaderboard:** N/A
- **Point of Contact:** Dan Hendrycks

### Dataset Summary

The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of problems
from mathematics competitions, including the AMC 10, AMC 12, AIME, and more. 
Each problem in MATH has a full step-by-step solution, which can be used to teach
models to generate answer derivations and explanations. For MATH-Hard, only the 
hardest questions were kept (Level 5). 

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

[More Information Needed]

## Dataset Structure

### Data Instances

A data instance consists of a competition math problem and its step-by-step solution written in LaTeX and natural language. The step-by-step solution contains the final answer enclosed in LaTeX's `\boxed` tag.

An example from the dataset is:
```
{'problem': 'A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.',
 'level': 'Level 1',
 'type': 'Counting & Probability',
 'solution': 'The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$.'}
```

### Data Fields

* `problem`: The competition math problem.
* `solution`: The step-by-step solution.
* `level`: We only kept tasks tagged as 'Level 5', the hardest level for the dataset.
* `type`: The subject of the problem: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra and Precalculus.


### Licensing Information

https://github.com/hendrycks/math/blob/main/LICENSE

### Citation Information

```bibtex
@article{hendrycksmath2021,
    title={Measuring Mathematical Problem Solving With the MATH Dataset},
    author={Dan Hendrycks
    and Collin Burns
    and Saurav Kadavath
    and Akul Arora
    and Steven Basart
    and Eric Tang
    and Dawn Song
    and Jacob Steinhardt},
    journal={arXiv preprint arXiv:2103.03874},
    year={2021}
}
```
