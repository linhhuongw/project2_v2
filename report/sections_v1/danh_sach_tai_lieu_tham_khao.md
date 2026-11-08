# Danh sách tài liệu tham khảo

Khai báo BibTeX của các tài liệu được trích dẫn trong báo cáo, theo mẫu trong `report/template.md`. Khi chuyển sang LaTeX, dán nội dung khối dưới vào file `.bib` và trích dẫn bằng `\cite{<khóa>}`.

| Trích dẫn trong bài | Khóa BibTeX |
|---|---|
| Baesens et al. (2003) | `baesens2003benchmarking` |
| Baesens et al. (2026) | `baesens2026foundation` |
| Breiman (2001) | `breiman2001random` |
| Bücker et al. (2022) | `bucker2022transparency` |
| Chen & Guestrin (2016) | `chen2016xgboost` |
| Cover & Hart (1967) | `cover1967nearest` |
| Cox (1958) | `cox1958regression` |
| Demšar (2006) | `demsar2006statistical` |
| Dorogush et al. (2018) | `dorogush2018catboost` |
| Dumitrescu et al. (2022) | `dumitrescu2022machine` |
| Friedman (2001) | `friedman2001greedy` |
| Grömping (2019) | `gromping2019south` |
| Gunnarsson et al. (2021) | `gunnarsson2021deep` |
| Hand & Henley (1997) | `hand1997statistical` |
| Hand (2006) | `hand2006classifier` |
| Henley & Hand (1996) | `henley1996knearest` |
| Ke et al. (2017) | `ke2017lightgbm` |
| Lessmann et al. (2015) | `lessmann2015benchmarking` |
| Maron (1961) | `maron1961automatic` |
| Prokhorenkova et al. (2018) | `prokhorenkova2018catboost` |
| Siddiqi (2017) | `siddiqi2017intelligent` |
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

@article{baesens2003benchmarking,
  title={Benchmarking state-of-the-art classification algorithms for credit scoring},
  author={Baesens, Bart and Van Gestel, Tony and Viaene, Stijn and Stepanova, Maria and Suykens, Johan and Vanthienen, Jan},
  journal={Journal of the Operational Research Society},
  volume={54},
  number={6},
  pages={627--635},
  year={2003},
  doi={10.1057/palgrave.jors.2601545}
}

@article{bucker2022transparency,
  title={Transparency, auditability, and explainability of machine learning models in credit scoring},
  author={B{\"u}cker, Michael and Szepannek, Gero and Gosiewska, Alicja and Biecek, Przemyslaw},
  journal={Journal of the Operational Research Society},
  volume={73},
  number={1},
  pages={70--90},
  year={2022},
  doi={10.1080/01605682.2021.1922098}
}

@article{dumitrescu2022machine,
  title={Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects},
  author={Dumitrescu, Elena and Hu{\'e}, Sullivan and Hurlin, Christophe and Tokpavi, Sessi},
  journal={European Journal of Operational Research},
  volume={297},
  number={3},
  pages={1178--1192},
  year={2022},
  doi={10.1016/j.ejor.2021.06.053}
}

@article{gunnarsson2021deep,
  title={Deep learning for credit scoring: Do or don't?},
  author={Gunnarsson, Bj{\"o}rn Rafn and vanden Broucke, Seppe and Baesens, Bart and {\'O}skarsd{\'o}ttir, Mar{\'i}a and Lemahieu, Wilfried},
  journal={European Journal of Operational Research},
  volume={295},
  number={1},
  pages={292--305},
  year={2021},
  doi={10.1016/j.ejor.2021.03.006}
}

@article{hand2006classifier,
  title={Classifier technology and the illusion of progress},
  author={Hand, David J.},
  journal={Statistical Science},
  volume={21},
  number={1},
  pages={1--14},
  year={2006},
  doi={10.1214/088342306000000060}
}

@article{hand1997statistical,
  title={Statistical classification methods in consumer credit scoring: a review},
  author={Hand, David J. and Henley, William E.},
  journal={Journal of the Royal Statistical Society: Series A (Statistics in Society)},
  volume={160},
  number={3},
  pages={523--541},
  year={1997},
  doi={10.1111/j.1467-985X.1997.00078.x}
}

@article{henley1996knearest,
  title={A k-nearest-neighbour classifier for assessing consumer credit risk},
  author={Henley, William E. and Hand, David J.},
  journal={Journal of the Royal Statistical Society: Series D (The Statistician)},
  volume={45},
  number={1},
  pages={77--95},
  year={1996},
  doi={10.2307/2348414}
}

@book{siddiqi2017intelligent,
  title={Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards},
  author={Siddiqi, Naeem},
  edition={2},
  publisher={John Wiley \& Sons},
  address={Hoboken, NJ},
  year={2017}
}
```
