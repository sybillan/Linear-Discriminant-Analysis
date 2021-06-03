Linear Discriminant Analysis
================

Linear discriminant analysis (LDA) and the related Fisher’s linear
discriminant are methods used in statistics, pattern recognition and
machine learning to find a linear combination of features which
characterizes or separates two or more classes of objects or events. The
resulting combination may be used as a linear classifier, or, more
commonly, for dimensionality reduction before later classification.

LDA assumes that predictors are normally distributed (Gaussian
distribution) and that the different classes have class-specific means
and equal variance/covariance.

The linear discriminant analysis can be easily computed using the
function lda() \[MASS package\].

``` r
library(datasets)
library(MASS)
library(ggplot2)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:MASS':
    ## 
    ##     select

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

Linear Discriminant Analysis, or LDA, uses the information from features
to create a new axis and projects the data on to the new axis in such a
way as to minimizes the variance and maximizes the distance between the
means of the classes.

``` r
summary(iris)
```

    ##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
    ##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
    ##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
    ##  Median :5.800   Median :3.000   Median :4.350   Median :1.300  
    ##  Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
    ##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
    ##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
    ##        Species  
    ##  setosa    :50  
    ##  versicolor:50  
    ##  virginica :50  
    ##                 
    ##                 
    ## 

``` r
lda.fit<-lda(Species~.,data=iris)
```

Performing Linear Discriminant Analysis on Iris dataset for
classification per Species. LDA determines group means and computes, for
each individual, the probability of belonging to the different groups.
The individual is then affected to the group with the highest
probability score.

The lda() outputs contain the following elements:

Prior probabilities of groups: the proportion of training observations
in each group. For example, there are 33% of the training observations
in the setosa group Group means: group center of gravity. Average of
each predictor within each class.

Coefficients of linear discriminants: Shows the linear combination of
predictor variables that are used to form the LDA decision rule.

for example, LD1 = 0.82*Sepal.Length + 1.53*Sepal.Width -
2.2*Petal.Length - 2.8*Petal.Width.

Similarly, LD2 = 0.024*Sepal.Length + 2.164*Sepal.Width -
0.93*Petal.Length +2.83*Petal.Width.

The “proportion of trace” returned by the lda() function is the
percentage separation achieved by each discriminant function. (99.12%
and 0.88%).

``` r
lda.fit
```

    ## Call:
    ## lda(Species ~ ., data = iris)
    ## 
    ## Prior probabilities of groups:
    ##     setosa versicolor  virginica 
    ##  0.3333333  0.3333333  0.3333333 
    ## 
    ## Group means:
    ##            Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## setosa            5.006       3.428        1.462       0.246
    ## versicolor        5.936       2.770        4.260       1.326
    ## virginica         6.588       2.974        5.552       2.026
    ## 
    ## Coefficients of linear discriminants:
    ##                     LD1         LD2
    ## Sepal.Length  0.8293776  0.02410215
    ## Sepal.Width   1.5344731  2.16452123
    ## Petal.Length -2.2012117 -0.93192121
    ## Petal.Width  -2.8104603  2.83918785
    ## 
    ## Proportion of trace:
    ##    LD1    LD2 
    ## 0.9912 0.0088

``` r
plot(lda.fit)
```

![](LDA_files/figure-gfm/lda2-1.png)<!-- -->

``` r
predictions <- lda.fit %>% predict(iris)
iris <- cbind(iris, predict(lda.fit)$x)

ggplot(iris, aes(LD1, LD2)) +
  geom_point(aes(color = Species))
```

![](LDA_files/figure-gfm/lda2-2.png)<!-- -->

``` r
predict(lda.fit,iris)
```

    ## $class
    ##   [1] setosa     setosa     setosa     setosa     setosa     setosa    
    ##   [7] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [13] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [19] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [25] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [31] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [37] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [43] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [49] setosa     setosa     versicolor versicolor versicolor versicolor
    ##  [55] versicolor versicolor versicolor versicolor versicolor versicolor
    ##  [61] versicolor versicolor versicolor versicolor versicolor versicolor
    ##  [67] versicolor versicolor versicolor versicolor virginica  versicolor
    ##  [73] versicolor versicolor versicolor versicolor versicolor versicolor
    ##  [79] versicolor versicolor versicolor versicolor versicolor virginica 
    ##  [85] versicolor versicolor versicolor versicolor versicolor versicolor
    ##  [91] versicolor versicolor versicolor versicolor versicolor versicolor
    ##  [97] versicolor versicolor versicolor versicolor virginica  virginica 
    ## [103] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [109] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [115] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [121] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [127] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [133] virginica  versicolor virginica  virginica  virginica  virginica 
    ## [139] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [145] virginica  virginica  virginica  virginica  virginica  virginica 
    ## Levels: setosa versicolor virginica
    ## 
    ## $posterior
    ##           setosa   versicolor    virginica
    ## 1   1.000000e+00 3.896358e-22 2.611168e-42
    ## 2   1.000000e+00 7.217970e-18 5.042143e-37
    ## 3   1.000000e+00 1.463849e-19 4.675932e-39
    ## 4   1.000000e+00 1.268536e-16 3.566610e-35
    ## 5   1.000000e+00 1.637387e-22 1.082605e-42
    ## 6   1.000000e+00 3.883282e-21 4.566540e-40
    ## 7   1.000000e+00 1.113469e-18 2.302608e-37
    ## 8   1.000000e+00 3.877586e-20 1.074496e-39
    ## 9   1.000000e+00 1.902813e-15 9.482936e-34
    ## 10  1.000000e+00 1.111803e-18 2.724060e-38
    ## 11  1.000000e+00 1.185277e-23 3.237084e-44
    ## 12  1.000000e+00 1.621649e-18 1.833201e-37
    ## 13  1.000000e+00 1.459225e-18 3.262506e-38
    ## 14  1.000000e+00 1.117219e-19 1.316642e-39
    ## 15  1.000000e+00 5.487399e-30 1.531265e-52
    ## 16  1.000000e+00 1.261505e-27 2.268705e-48
    ## 17  1.000000e+00 6.754338e-25 3.868271e-45
    ## 18  1.000000e+00 4.223741e-21 1.224313e-40
    ## 19  1.000000e+00 1.774911e-22 2.552153e-42
    ## 20  1.000000e+00 2.593237e-22 5.792079e-42
    ## 21  1.000000e+00 1.274639e-19 4.357774e-39
    ## 22  1.000000e+00 1.465999e-20 1.987241e-39
    ## 23  1.000000e+00 6.569280e-25 7.769177e-46
    ## 24  1.000000e+00 8.912348e-15 9.178624e-32
    ## 25  1.000000e+00 1.070702e-15 1.167516e-33
    ## 26  1.000000e+00 2.497339e-16 5.710269e-35
    ## 27  1.000000e+00 3.967732e-17 4.378624e-35
    ## 28  1.000000e+00 1.548165e-21 1.595360e-41
    ## 29  1.000000e+00 9.271847e-22 6.297955e-42
    ## 30  1.000000e+00 9.665144e-17 2.977974e-35
    ## 31  1.000000e+00 2.299936e-16 7.182666e-35
    ## 32  1.000000e+00 1.975404e-19 2.788334e-38
    ## 33  1.000000e+00 7.100041e-27 2.216408e-48
    ## 34  1.000000e+00 1.610295e-28 2.743783e-50
    ## 35  1.000000e+00 1.205219e-17 1.277245e-36
    ## 36  1.000000e+00 1.597186e-21 9.033772e-42
    ## 37  1.000000e+00 1.939869e-24 1.662808e-45
    ## 38  1.000000e+00 3.310234e-23 7.004971e-44
    ## 39  1.000000e+00 4.190242e-17 6.991441e-36
    ## 40  1.000000e+00 1.769359e-20 3.541694e-40
    ## 41  1.000000e+00 1.063014e-21 2.003866e-41
    ## 42  1.000000e+00 2.174217e-11 1.213781e-28
    ## 43  1.000000e+00 1.540753e-18 1.305719e-37
    ## 44  1.000000e+00 8.940589e-16 1.315511e-32
    ## 45  1.000000e+00 1.616206e-17 3.205992e-35
    ## 46  1.000000e+00 1.714743e-16 7.172435e-35
    ## 47  1.000000e+00 2.083089e-22 2.289783e-42
    ## 48  1.000000e+00 2.793482e-18 2.629539e-37
    ## 49  1.000000e+00 2.597560e-23 9.820820e-44
    ## 50  1.000000e+00 2.322258e-20 4.241757e-40
    ## 51  1.969732e-18 9.998894e-01 1.105878e-04
    ## 52  1.242878e-19 9.992575e-01 7.425297e-04
    ## 53  2.088263e-22 9.958069e-01 4.193053e-03
    ## 54  2.198898e-22 9.996423e-01 3.576502e-04
    ## 55  4.213678e-23 9.955903e-01 4.409655e-03
    ## 56  8.127287e-23 9.985020e-01 1.497982e-03
    ## 57  3.549900e-22 9.858346e-01 1.416542e-02
    ## 58  5.007065e-14 9.999999e-01 1.119811e-07
    ## 59  5.683334e-20 9.998781e-01 1.218649e-04
    ## 60  1.241039e-20 9.995027e-01 4.973085e-04
    ## 61  1.956628e-18 9.999986e-01 1.420841e-06
    ## 62  5.968900e-20 9.992294e-01 7.705716e-04
    ## 63  2.716128e-18 9.999988e-01 1.220169e-06
    ## 64  1.184445e-23 9.943267e-01 5.673286e-03
    ## 65  5.574931e-14 9.999984e-01 1.649215e-06
    ## 66  2.369511e-17 9.999573e-01 4.268212e-05
    ## 67  8.429328e-24 9.806471e-01 1.935289e-02
    ## 68  2.505072e-16 9.999991e-01 9.151716e-07
    ## 69  1.670352e-27 9.595735e-01 4.042653e-02
    ## 70  1.341503e-17 9.999967e-01 3.296105e-06
    ## 71  7.408118e-28 2.532282e-01 7.467718e-01
    ## 72  9.399292e-17 9.999907e-01 9.345291e-06
    ## 73  7.674672e-29 8.155328e-01 1.844672e-01
    ## 74  2.683018e-22 9.995723e-01 4.277469e-04
    ## 75  7.813875e-18 9.999758e-01 2.421458e-05
    ## 76  2.073207e-18 9.999171e-01 8.290530e-05
    ## 77  6.357538e-23 9.982541e-01 1.745936e-03
    ## 78  5.639473e-27 6.892131e-01 3.107869e-01
    ## 79  3.773528e-23 9.925169e-01 7.483138e-03
    ## 80  9.555338e-12 1.000000e+00 1.910624e-08
    ## 81  1.022109e-17 9.999970e-01 3.007748e-06
    ## 82  9.648075e-16 9.999997e-01 3.266704e-07
    ## 83  1.616405e-16 9.999962e-01 3.778441e-06
    ## 84  4.241952e-32 1.433919e-01 8.566081e-01
    ## 85  1.724514e-24 9.635576e-01 3.644242e-02
    ## 86  1.344746e-20 9.940401e-01 5.959931e-03
    ## 87  3.304868e-21 9.982223e-01 1.777672e-03
    ## 88  2.034571e-23 9.994557e-01 5.443096e-04
    ## 89  5.806986e-18 9.999486e-01 5.137101e-05
    ## 90  5.981190e-21 9.998183e-01 1.816870e-04
    ## 91  5.878614e-23 9.993856e-01 6.144200e-04
    ## 92  5.399006e-22 9.980934e-01 1.906591e-03
    ## 93  3.559507e-18 9.999887e-01 1.128570e-05
    ## 94  2.104146e-14 9.999999e-01 1.135016e-07
    ## 95  4.700877e-21 9.996980e-01 3.020226e-04
    ## 96  1.584328e-17 9.999817e-01 1.826327e-05
    ## 97  2.802293e-19 9.998892e-01 1.108315e-04
    ## 98  1.626918e-18 9.999536e-01 4.640488e-05
    ## 99  7.638378e-11 1.000000e+00 1.867332e-08
    ## 100 4.679301e-19 9.999269e-01 7.305863e-05
    ## 101 7.503075e-52 7.127303e-09 1.000000e+00
    ## 102 5.213802e-38 1.078251e-03 9.989217e-01
    ## 103 1.231264e-42 2.592826e-05 9.999741e-01
    ## 104 1.537499e-38 1.068139e-03 9.989319e-01
    ## 105 6.242501e-46 1.812963e-06 9.999982e-01
    ## 106 4.209281e-49 6.656263e-07 9.999993e-01
    ## 107 3.797837e-33 4.862025e-02 9.513797e-01
    ## 108 1.352176e-42 1.395463e-04 9.998605e-01
    ## 109 1.323390e-42 2.235313e-04 9.997765e-01
    ## 110 3.453358e-46 1.727277e-07 9.999998e-01
    ## 111 5.452660e-32 1.305353e-02 9.869465e-01
    ## 112 1.182560e-37 1.673875e-03 9.983261e-01
    ## 113 5.204321e-39 2.006352e-04 9.997994e-01
    ## 114 1.269953e-40 1.948672e-04 9.998051e-01
    ## 115 1.685361e-45 1.000455e-06 9.999990e-01
    ## 116 5.141640e-40 2.605493e-05 9.999739e-01
    ## 117 1.909820e-35 6.083553e-03 9.939164e-01
    ## 118 1.207799e-44 1.503799e-06 9.999985e-01
    ## 119 3.181265e-59 1.317279e-09 1.000000e+00
    ## 120 1.598511e-33 2.207990e-01 7.792010e-01
    ## 121 1.119461e-42 6.451865e-06 9.999935e-01
    ## 122 3.038170e-37 8.272676e-04 9.991727e-01
    ## 123 6.032879e-50 9.509838e-07 9.999990e-01
    ## 124 1.951261e-31 9.711942e-02 9.028806e-01
    ## 125 1.956408e-39 8.836845e-05 9.999116e-01
    ## 126 1.109337e-36 2.679310e-03 9.973207e-01
    ## 127 7.841997e-30 1.883675e-01 8.116325e-01
    ## 128 7.964690e-30 1.342431e-01 8.657569e-01
    ## 129 6.190641e-44 1.303681e-05 9.999870e-01
    ## 130 1.406448e-32 1.036823e-01 8.963177e-01
    ## 131 4.108129e-42 1.442338e-04 9.998558e-01
    ## 132 1.555697e-36 5.198047e-04 9.994802e-01
    ## 133 1.320330e-45 3.014091e-06 9.999970e-01
    ## 134 1.283891e-28 7.293881e-01 2.706119e-01
    ## 135 1.926560e-35 6.602253e-02 9.339775e-01
    ## 136 1.271083e-45 2.152818e-06 9.999978e-01
    ## 137 3.038963e-44 8.881859e-07 9.999991e-01
    ## 138 4.605973e-35 6.165648e-03 9.938344e-01
    ## 139 4.538634e-29 1.925262e-01 8.074738e-01
    ## 140 2.140232e-36 8.290895e-04 9.991709e-01
    ## 141 6.570902e-45 1.180810e-06 9.999988e-01
    ## 142 6.202588e-36 4.276398e-04 9.995724e-01
    ## 143 5.213802e-38 1.078251e-03 9.989217e-01
    ## 144 1.073945e-45 1.028519e-06 9.999990e-01
    ## 145 4.048249e-46 2.524984e-07 9.999997e-01
    ## 146 4.970070e-39 7.473361e-05 9.999253e-01
    ## 147 4.616611e-36 5.898784e-03 9.941012e-01
    ## 148 5.548962e-35 3.145874e-03 9.968541e-01
    ## 149 1.613687e-40 1.257468e-05 9.999874e-01
    ## 150 2.858012e-33 1.754229e-02 9.824577e-01
    ## 
    ## $x
    ##            LD1          LD2
    ## 1    8.0617998  0.300420621
    ## 2    7.1286877 -0.786660426
    ## 3    7.4898280 -0.265384488
    ## 4    6.8132006 -0.670631068
    ## 5    8.1323093  0.514462530
    ## 6    7.7019467  1.461720967
    ## 7    7.2126176  0.355836209
    ## 8    7.6052935 -0.011633838
    ## 9    6.5605516 -1.015163624
    ## 10   7.3430599 -0.947319209
    ## 11   8.3973865  0.647363392
    ## 12   7.2192969 -0.109646389
    ## 13   7.3267960 -1.072989426
    ## 14   7.5724707 -0.805464137
    ## 15   9.8498430  1.585936985
    ## 16   9.1582389  2.737596471
    ## 17   8.5824314  1.834489452
    ## 18   7.7807538  0.584339407
    ## 19   8.0783588  0.968580703
    ## 20   8.0209745  1.140503656
    ## 21   7.4968023 -0.188377220
    ## 22   7.5864812  1.207970318
    ## 23   8.6810429  0.877590154
    ## 24   6.2514036  0.439696367
    ## 25   6.5589334 -0.389222752
    ## 26   6.7713832 -0.970634453
    ## 27   6.8230803  0.463011612
    ## 28   7.9246164  0.209638715
    ## 29   7.9912902  0.086378713
    ## 30   6.8294645 -0.544960851
    ## 31   6.7589549 -0.759002759
    ## 32   7.3749525  0.565844592
    ## 33   9.1263463  1.224432671
    ## 34   9.4676820  1.825226345
    ## 35   7.0620139 -0.663400423
    ## 36   7.9587624 -0.164961722
    ## 37   8.6136720  0.403253602
    ## 38   8.3304176  0.228133530
    ## 39   6.9341201 -0.705519379
    ## 40   7.6882313 -0.009223623
    ## 41   7.9179372  0.675121313
    ## 42   5.6618807 -1.934355243
    ## 43   7.2410147 -0.272615132
    ## 44   6.4144356  1.247301306
    ## 45   6.8594438  1.051653957
    ## 46   6.7647039 -0.505151855
    ## 47   8.0818994  0.763392750
    ## 48   7.1867690 -0.360986823
    ## 49   8.3144488  0.644953177
    ## 50   7.6719674 -0.134893840
    ## 51  -1.4592755  0.028543764
    ## 52  -1.7977057  0.484385502
    ## 53  -2.4169489 -0.092784031
    ## 54  -2.2624735 -1.587252508
    ## 55  -2.5486784 -0.472204898
    ## 56  -2.4299673 -0.966132066
    ## 57  -2.4484846  0.795961954
    ## 58  -0.2226665 -1.584673183
    ## 59  -1.7502012 -0.821180130
    ## 60  -1.9584224 -0.351563753
    ## 61  -1.1937603 -2.634455704
    ## 62  -1.8589257  0.319006544
    ## 63  -1.1580939 -2.643409913
    ## 64  -2.6660572 -0.642504540
    ## 65  -0.3783672  0.086638931
    ## 66  -1.2011726  0.084437359
    ## 67  -2.7681025  0.032199536
    ## 68  -0.7768540 -1.659161847
    ## 69  -3.4980543 -1.684956162
    ## 70  -1.0904279 -1.626583496
    ## 71  -3.7158961  1.044514421
    ## 72  -0.9976104 -0.490530602
    ## 73  -3.8352593 -1.405958061
    ## 74  -2.2574125 -1.426794234
    ## 75  -1.2557133 -0.546424197
    ## 76  -1.4375576 -0.134424979
    ## 77  -2.4590614 -0.935277280
    ## 78  -3.5184849  0.160588866
    ## 79  -2.5897987 -0.174611728
    ## 80   0.3074879 -1.318871459
    ## 81  -1.1066918 -1.752253714
    ## 82  -0.6055246 -1.942980378
    ## 83  -0.8987038 -0.904940034
    ## 84  -4.4984664 -0.882749915
    ## 85  -2.9339780  0.027379106
    ## 86  -2.1036082  1.191567675
    ## 87  -2.1425821  0.088779781
    ## 88  -2.4794560 -1.940739273
    ## 89  -1.3255257 -0.162869550
    ## 90  -1.9555789 -1.154348262
    ## 91  -2.4015702 -1.594583407
    ## 92  -2.2924888 -0.332860296
    ## 93  -1.2722722 -1.214584279
    ## 94  -0.2931761 -1.798715092
    ## 95  -2.0059888 -0.905418042
    ## 96  -1.1816631 -0.537570242
    ## 97  -1.6161564 -0.470103580
    ## 98  -1.4215888 -0.551244626
    ## 99   0.4759738 -0.799905482
    ## 100 -1.5494826 -0.593363582
    ## 101 -7.8394740  2.139733449
    ## 102 -5.5074800 -0.035813989
    ## 103 -6.2920085  0.467175777
    ## 104 -5.6054563 -0.340738058
    ## 105 -6.8505600  0.829825394
    ## 106 -7.4181678 -0.173117995
    ## 107 -4.6779954 -0.499095015
    ## 108 -6.3169269 -0.968980756
    ## 109 -6.3277368 -1.383289934
    ## 110 -6.8528134  2.717589632
    ## 111 -4.4407251  1.347236918
    ## 112 -5.4500957 -0.207736942
    ## 113 -5.6603371  0.832713617
    ## 114 -5.9582372 -0.094017545
    ## 115 -6.7592628  1.600232061
    ## 116 -5.8070433  2.010198817
    ## 117 -5.0660123 -0.026273384
    ## 118 -6.6088188  1.751635872
    ## 119 -9.1714749 -0.748255067
    ## 120 -4.7645357 -2.155737197
    ## 121 -6.2728391  1.649481407
    ## 122 -5.3607119  0.646120732
    ## 123 -7.5811998 -0.980722934
    ## 124 -4.3715028 -0.121297458
    ## 125 -5.7231753  1.293275530
    ## 126 -5.2791592 -0.042458238
    ## 127 -4.0808721  0.185936572
    ## 128 -4.0770364  0.523238483
    ## 129 -6.5191040  0.296976389
    ## 130 -4.5837194 -0.856815813
    ## 131 -6.2282401 -0.712719638
    ## 132 -5.2204877  1.468195094
    ## 133 -6.8001500  0.580895175
    ## 134 -3.8151597 -0.942985932
    ## 135 -5.1074897 -2.130589999
    ## 136 -6.7967163  0.863090395
    ## 137 -6.5244960  2.445035271
    ## 138 -4.9955028  0.187768525
    ## 139 -3.9398530  0.614020389
    ## 140 -5.2038309  1.144768076
    ## 141 -6.6530868  1.805319760
    ## 142 -5.1055595  1.992182010
    ## 143 -5.5074800 -0.035813989
    ## 144 -6.7960192  1.460686950
    ## 145 -6.8473594  2.428950671
    ## 146 -5.6450035  1.677717335
    ## 147 -5.1795646 -0.363475041
    ## 148 -4.9677409  0.821140550
    ## 149 -5.8861454  2.345090513
    ## 150 -4.6831543  0.332033811

Predict() function returns a list with three elements: class, which
class the observation belongs to, posterior probability that the
observation belongs to kth class and x that contains the linear
discriminants.

References:

<https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2#>:\~:text=Linear%20Discriminant%20Analysis%20(LDA)%20is,as%20much%20information%20as%20possible.

<https://www.displayr.com/linear-discriminant-analysis-in-r-an-introduction/>

<http://www.sthda.com/english/articles/36-classification-methods-essentials/146-discriminant-analysis-essentials-in-r/>

James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani.
2014. An Introduction to Statistical Learning: With Applications in R.
Springer Publishing Company, Incorporated.
