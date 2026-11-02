# Danh sách tài liệu tham khảo

Khai báo BibTeX của các tài liệu được trích dẫn trong báo cáo, theo mẫu trong `report/template.md`. Khi chuyển sang LaTeX, dán nội dung khối dưới vào file `.bib` và trích dẫn bằng `\cite{<khóa>}`.

| Trích dẫn trong bài | Khóa BibTeX |
|---|---|
| Baesens et al. (2026) | `baesens2026foundation` |
| Breiman (2001) | `breiman2001random` |
| Chen & Guestrin (2016) | `chen2016xgboost` |
| Cover & Hart (1967) | `cover1967nearest` |
| Cox (1958) | `cox1958regression` |
| Demšar (2006) | `demsar2006statistical` |
| Dorogush et al. (2018) | `dorogush2018catboost` |
| Friedman (2001) | `friedman2001greedy` |
| Grömping (2019) | `gromping2019south` |
| Ke et al. (2017) | `ke2017lightgbm` |
| Lessmann et al. (2015) | `lessmann2015benchmarking` |
| Maron (1961) | `maron1961automatic` |
| Prokhorenkova et al. (2018) | `prokhorenkova2018catboost` |
| Yeh & Lien (2009) | `yeh2009comparisons` |

```
@article{baesens2026foundation,
  title={Foundation Models for Credit Risk Prediction: A Game Changer?},
  author={Baesens, Bart and Goethals, Alexander and Lessmann, Stefan and De Vos, Simon and Bravo, Cristi{\'a}n and Martens, David and Medina-Olivares, Victor and Mues, Christophe and {\'O}skarsd{\'o}ttir, Mar{\'i}a and vanden Broucke, Seppe and Verdonck, Tim and Verbeke, Wouter},
  journal={arXiv preprint arXiv:2605.18147},
  year={2026}
}

@article{breiman2001random,
  title={Random forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001},
  publisher={Kluwer Academic Publishers}
}

@inproceedings{chen2016xgboost,
  title={{XGBoost}: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016},
  organization={ACM},
  doi={10.1145/2939672.2939785}
}

@article{cover1967nearest,
  title={Nearest neighbor pattern classification},
  author={Cover, Thomas M and Hart, Peter E},
  journal={IEEE Transactions on Information Theory},
  volume={13},
  number={1},
  pages={21--27},
  year={1967},
  publisher={IEEE}
}

@article{cox1958regression,
  title={The regression analysis of binary sequences},
  author={Cox, David R},
  journal={Journal of the Royal Statistical Society: Series B (Methodological)},
  volume={20},
  number={2},
  pages={215--242},
  year={1958}
}

@article{demsar2006statistical,
  title={Statistical comparisons of classifiers over multiple data sets},
  author={Dem{\v{s}}ar, Janez},
  journal={Journal of Machine Learning Research},
  volume={7},
  pages={1--30},
  year={2006}
}

@article{dorogush2018catboost,
  title={{CatBoost}: gradient boosting with categorical features support},
  author={Dorogush, Anna Veronika and Ershov, Vasily and Gulin, Andrey},
  journal={arXiv preprint arXiv:1810.11363},
  year={2018}
}

@article{friedman2001greedy,
  title={Greedy function approximation: A gradient boosting machine},
  author={Friedman, Jerome H},
  journal={The Annals of Statistics},
  volume={29},
  number={5},
  pages={1189--1232},
  year={2001},
  publisher={Institute of Mathematical Statistics}
}

@techreport{gromping2019south,
  title={South German Credit Data: Correcting a Widely Used Data Set},
  author={Gr{\"o}mping, Ulrike},
  institution={Beuth University of Applied Sciences Berlin},
  type={Reports in Mathematics, Physics and Chemistry},
  number={4/2019},
  year={2019}
}

@inproceedings{ke2017lightgbm,
  title={{LightGBM}: A highly efficient gradient boosting decision tree},
  author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems 30 (NIPS 2017)},
  pages={3146--3154},
  year={2017}
}

@article{lessmann2015benchmarking,
  title={Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research},
  author={Lessmann, Stefan and Baesens, Bart and Seow, Hsin-Vonn and Thomas, Lyn C},
  journal={European Journal of Operational Research},
  volume={247},
  number={1},
  pages={124--136},
  year={2015},
  publisher={Elsevier}
}

@article{maron1961automatic,
  title={Automatic indexing: An experimental inquiry},
  author={Maron, Melvin E},
  journal={Journal of the ACM},
  volume={8},
  number={3},
  pages={404--417},
  year={1961},
  publisher={ACM}
}

@inproceedings{prokhorenkova2018catboost,
  title={{CatBoost}: unbiased boosting with categorical features},
  author={Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
  booktitle={Advances in Neural Information Processing Systems 31 (NeurIPS 2018)},
  pages={6638--6648},
  year={2018}
}

@article{yeh2009comparisons,
  title={The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients},
  author={Yeh, I-Cheng and Lien, Che-hui},
  journal={Expert Systems with Applications},
  volume={36},
  number={2},
  pages={2473--2480},
  year={2009},
  publisher={Elsevier}
}
```
