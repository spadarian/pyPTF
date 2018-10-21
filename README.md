# Welcome to pyPTF!
[![Build Status](https://travis-ci.org/spadarian/pyPTF.svg?branch=master)](https://travis-ci.org/spadarian/pyPTF)

This is a framework to develop [pedotransfer functions](https://en.wikipedia.org/wiki/Pedotransfer_function) using genetic programming.

A pedotransfer function is a model to predict soil properties based on other, easier to measure, soil properties. Traditionally, pedotransfer functions are represented as expressions (formulas) and that is why [symbolic regression](https://en.wikipedia.org/wiki/Symbolic_regression) is a great alternative to develop them.

This project wouldn't be possible without the great [glplearn](https://github.com/trevorstephens/gplearn) library, which implements Symbolic regression. In Trevor's words:
> Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression that best describes a relationship. It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets in order to predict new data. Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations.

## Uncertainty

Natural systems are complex and every model that tries to represent them have uncertainties. This library implements a method to represent and report this uncertainty based on [fuzzy clustering](https://en.wikipedia.org/wiki/Fuzzy_clustering).
