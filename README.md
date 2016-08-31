An old fork of [OpenGM](https://github.com/opengm/opengm) with an implementation of an inference algorithm for second order
graphical models. The algorithm is described in [qpdc.pdf](https://github.com/pesser/opengm/blob/master/qpdc.pdf) which I wrote
for a seminar of the [Image & Pattern Analysis Group](http://ipa.math.uni-heidelberg.de/) at the
[IWR](https://www.iwr.uni-heidelberg.de/).
The algorithm uses a specific representation of the objective function in terms of a difference of two convex functions to
optimize it. Based on the evaluation in the aforementioned document, it was not included in the library which
[evolved](https://github.com/opengm/opengm) without this algorithm. This repository is kept as an archive.
