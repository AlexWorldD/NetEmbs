# NetEmbs
[![Build Status](https://travis-ci.com/AlexWorldD/NetEmbs.svg?token=KxxnGy2fzypoq5mv4Y2J&branch=master)](https://travis-ci.com/AlexWorldD/NetEmbs) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
### TODO before next meeting
- [ ] Mathematical definition of node's context. Compare it with context definition for words
- [ ] Modify RandomWalk procedure according to new definition of context
- [ ] Fix simulation code
### Initial steps
- [x] Install *simpy*. Fix Marcel's model. Get sample dataset
- [x] Install *networkx*, play with it.
- [x] Implement split for debit/credit and normalization functions.
- [x] Implement basic visualisation
- [x] Implement Neighborhoods functions: IN/OUT edges.
- [x] Implement visualisation of neighbors (highlight IN/OUT context).
- [ ] Define the architecture of Python module
### Questions (15.02.2019)
- [x] Is it possible that the same financial account could be debited during one set of BPs and credited during another set of BP? // **YES**
-----
## Literature
1. Boersma M. et al. Financial statement networks: an application of network theory in the audit // Draft of paper. 2019. P. 1–33.
2. Perozzi B., Al-Rfou R., Skiena S. DeepWalk: Online Learning of Social Representations. 2014.
Grover A., Leskovec J. node2vec // Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16. New York, New York, USA: ACM Press, 2016. P. 855–864.
3. Dong Y., Chawla N. V., Swami A. metapath2vec // Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’17. New York, New York, USA: ACM Press, 2017. P. 135–144
4. Gao M. et al. BiNE // The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval - SIGIR ’18. 2018. P. 715–724.

