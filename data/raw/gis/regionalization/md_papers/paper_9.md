# On the Performance of the Subtour Elimination Constraints Approach for the p-Regions Problem: A Computational Study

## Abstract
The p-regions is a mixed integer programming (MIP) model for the exhaustive clustering of a set of n geographic areas into p spatially contiguous regions while minimizing measures of intraregional heterogeneity. This is an NP-hard problem that requires a constant research of strategies to increase the size of instances that can be solved using exact optimization techniques. In this article, we explore the beneﬁts of an iterative process that begins by solving the relaxed version of the p-regions that removes the constraints that guarantee the spatial contiguity of the regions. Then, additional constraints are incorporated iteratively to solve spatial discontinuities in the regions. In particular we explore the relationship between the level of spatial autocorrelation of the aggregation variable and the beneﬁts obtained from this iterative process. The results show that high levels of spatial autocorrelation reduce computational times because the spatial patterns tend to create spatially contiguous regions. However, we found that the greatest beneﬁts are obtained in two situations: (1) when n=p /C21 3; and (2) when the parameter p is close to the number of clusters in the spatial pattern of the aggregation variable.

## Introduction
The p-regions model devised by Duque, Church, and Middleton (2011) is a mixed integer programming (MIP) model that solves problems related to the exhaustive clustering of a set of n geographic areas into p spatially contiguous regions while minimizing measures of intraregional heterogeneity. 1 Although problems of spatial clustering have been studied in the literature since the early 1960s (Vickrey 1961), the p-regions model is considered the ﬁrst MIP formulation of the spatial clustering problem, and its appearance made the way for a series of p-regions models such as the max-p-regions model (Duque, Anselin, and Rey 2012), the p-functional regions model (Kim, Chun, and Kim 2013), the p-compact regions model (Li, Church, and Goodchild 2014b), and the network- max-p-regions (She, Duque, and Ye 2017).

The spatial clustering problem is classiﬁed as nondeterministic polynomial-time hard ( NP-hard) (Cliff et al. 1975; Keane 1975), which implies that the complexity of the problem, and therefore the solution time, increases drastically as the problem size increases. Such complexities have positioned heuristic solutions as the most appropriate solution techniques for addressing large instances (Duque, Ramos, and Surinach 2007; Li, Church, and Goodchild 2014a). However, as is already common for many NP-hard problems, constant research must be conducted to increase the size of instances that can be solved using exact optimization techniques.

Duque, Church, and Middleton (2011) proposed three different types of constraints to guarantee that each region is contiguous. One of these strategies, which is the focus of this article, is called the Tree model, where contiguity is achieved by connecting the areas in a region through an acyclic set of links (a tree). The authors prevent cycles by adapting the tour-breaking constraints devised by Miller, Tucker, and Zemlin (1960) (MTZ) for the Travelling Salesman Problem (TSP). The same article Duque, Church, and Middleton (2011) then showed the potential of using the subtour elimination constraint proposed by Dantzig, Fulkerson, and Johnson (1954) (DFJ constraints). In this case, the Tree model is solved without the MTZ constraints, and the resulting subtours are prevented from reappearing by adding the corresponding DFJ constraints. This process is repeated until no subtours appear in the solution. The results obtained from a small computational experiment with 10 instances show the potential of this strategy, with an average reduction of running time of 76.93% and an increase of 10.0% in the number of instances solved to optimality.

In this article, we explore the strategy proposed by Duque, Church, and Middleton (2011) in greater depth. The present work is not focused on proposing a new approach to reduce the computational time but rather aims to obtain a better understanding of the complexity of the spatial clustering problem and of the behavior of the DFJ constraints for the p-regions model. We attempt to answer questions such as is there a relationship between the level of spatial autocorrelation of the aggregation variable and the beneﬁts obtained from the DFJ constraints? Intuitively, a higher level of spatial autocorrelation should correspond to a lower amount of the required DFJ constraints, because the spatial clusters would tend to emerge naturally. Some arguments in this vein, albeit without any computational evidence, have been presented by Openshaw and Wymer (1995) and Duque, Art/C19 ıs, and Ramos (2006) for non-exact spatial clustering. We also explore the relationship between the beneﬁts derived from the DFJ constraints and the parameter p ( n u m b e ro fr e g i o n s ) f o raﬁ x e dn u m b e ro fa r e a sn. In this regard, Duque, Church, and Middleton (2011) ﬁnd that the lower the ratio n/p, the greater the beneﬁts obtained from the DFJ constraints. We seek to test whether their ﬁnding holds for larger instances and different levels of spatial autocorrelation.

Obtaining a deeper understanding of this NP-hard problem will provide the academic community with new elements for solving larger instances not only of the p-regions model but also of other spatially constrained clustering models derived from the p-regions model. This also generates an interesting link between exploratory spatial data analysis (ESDA) and this MIP problem that can open the door to innovations in the area of spatially constrained clustering.

The remainder of the article is organized as follows. A literature review is presented in section “Literature review.” The exact formulation of the p-regions model is introduced in section “The exact formulation of the p-regions model— Tree version.” The solution approach is described in section “The subtour elimination constraints for the p-regions model,” and our computational experiments are presented in sections “Computational experiment” and “Disentangling the relationship between the DFJ-p-regions and rho.” Finally, conclusions are stated and avenues for future research are proposed in section “Conclusions.”

## Literature review
Problems related to aggregating areas into spatially contiguous regions have been given different names in the literature (e.g., region-building [Byfuglien and Nordgard 1973], conditional clustering [Lefkovitch 1980], clustering with relational constraints [Ferligoj and Batagelj 1982], constrained clustering [Legendre 1987], contiguity constrained clustering [Murtagh 1992], regional clustering [Maravalle and Simeone 1995], contiguity constrained classiﬁcation [Gordon 1996], regionalization [Wise, Haining, and Ma 1997], and clustering under connectivity constraints [Hansen et al. 2003]). All of these problems have two common factors: the spatial contiguity constraint and the exhaustive clustering. However, each problem includes additional conditions or characteristics depending on the area of application. For example, problems within the area of Electoral and School Districting impose additional constraints such as compactness or similarities to existing solutions (Browdy 1990; Macmillan and Pierce 1994; Williams 1995; Macmillan 2001; Caro et al. 2004; Ricca and Simeone 2008); other models of Sales Territory Alignment, Police Districting and Home Care Districting impose additional constraints to guarantee balanced workloads and minimum travel distances (Zoltners and Sinha 1983; D’amico et al. 2002; Blais, Lapierre, and Laporte 2003); and other models (e.g., Zone Design) seek to maximize the goodness of ﬁt of a statistical model or the correlation between two variables (Openshaw and Rao 1995).

Regarding solution methods, due to the complexities of these models, heuristic methods are the most frequently used solution approaches. These methods typically decompose a solution into two phases: (1) construction of an initial feasible solution, usually by growing regions from seed areas and (2) local searches, which involve iterative modiﬁcations of a feasible solution while optimizing an aggregation criterion. For a literature review on these models, see Murtagh (1985), Gordon (1996) and Duque, Ramos, and Surinach (2007). More recently, Laura et al. (2015) explored the potential of combining heuristic solutions with parallelized computing to solve large spatially constrained clustering problems efﬁciently.

The use of exact formulations involving explicit spatial contiguity constraints is a relatively recent trend in the literature. To the best of our knowledge, the ﬁrst MIP formulation to address the aggregation of n areas into p spatially contiguous regions was proposed by Duque, Church, and Middleton (2011) and is known as the p-regions problem. 2 In this work, the authors introduced three different strategies for ensuring contiguity: (1) an adaption of Miller, Tucker, and Zemlin (1960) tour-breaking constraints (or MTZ constraints) for the so-called Traveling Salesman Problem; (2) the use of ordered-area assignment variables based on an extension of an approach proposed by Cova and Church (2000) for the geographical site design problem; and (3) an extension of the work by Shirabe (2005) using ﬂow constraints. These strategies are referred to as Tree, Order, and Flow, respectively.

The publication of the p-regions model motivated research on exact formulations of MIP models that require contiguity constraints along with additional conditions. The max- p-region, devised by Duque, Anselin, and Rey (2012), aggregates n areas into the maximum number of spatially contiguous regions such that the value of a spatially extensive regional attribute is higher than a predeﬁned threshold value; the p-compact region, proposed by Li, Church, and Goodchild (2014b), seeks to aggregate n spatial units into p-compact, contiguous regions; the p-functional region, created by Kim, Chun, and Kim (2013), deﬁnes p functional regions by considering geographic ﬂows; and the network-max- p-regions, devised by She, Duque, and Ye (2017), takes into account the inﬂuence of a given street network to generate two types of regions: corridor regions, consisting on elongated regions along roads, and max- p-regions (i.e., shape-free regions) located either close to or far from roads.

The computational complexity associated with the contiguity constraints is one of the main challenges related to the treatment of spatially constrained clustering as an exact optimization problem. These constraints ensure that the areas assigned to the same region are spatially connected. One interesting aspect of this set of constraints is that it is too large, with only a fraction of these constraints actually required to guarantee optimality. Based on this characteristic, Duque, Church, and Middleton (2011) proposed a promising strategy to reduce the running time of the Tree version of the p-regions while guaranteeing optimality. Their proposal consisted of applying the subtour elimination constraint proposed by Dantzig, Fulkerson, and Johnson (1954) to solve the TSP problem. 3 Dantzig, Fulkerson, and Johnson (1954) introduced a means of solving the TSP in an iterative fashion. In their work, the authors present subtour elimination constraints as additional constraints on the TSP as a means of removing subtours that can arise when solving the problem in its relaxed form. They found that it is sufﬁcient to begin with the relaxed form of the TSP as long as it is possible to determine when the solution of the relaxed problem contains subtours and, if so, to ﬁnd these subtours in the current solution. They applied this idea to cope with the exponential number of constraints that arise when eliminating subtours. Their remarkable contribution served as a precursor to the development of the branch-and-cut algorithms that are used in practice to solve a broad range of optimization models (Gr €otschel and Nemhauser 2008). It also became the preferred approach to solve the TSP by integer programming (Orman and Williams 2006), and it has been successfully applied in other spatial optimization problems, such as the shortest covering path problem (Current, ReVelle, and Cohon 1984) and the Critical Cluster Problem (Church and Cova 2000).

In this article, we aim to explore the synergies that may result from combining the two research ﬁelds of exact optimization and ESDA (Anselin 2005; Ye and Rey 2013). The contribution by Duque, Church, and Middleton (2011) offers a good case study for this exploration. Specifically, we are interested in knowing how the spatial distribution of the aggregation variable (i.e., the variable used to measure the dissimilarity between the areas to be aggregated) can affect the complexity of the p-regions (in its Tree version) and the effectiveness of the subtour elimination constraints. Using aggregation variables with different levels of spatial autocorrelation, we can generate a range of spatial distributions that vary from a chessboard-like pattern (high negative spatial autocorrelation) to random spatial patterns (no spatial autocorrelation) and then to clustered spatial patterns (high spatial autocorrelation). By analyzing the performance of the subtour elimination constraints for the p-regions in these spatial patterns, we aim to be able to test whether high spatial autocorrelation levels can signiﬁcantly reduce the number of subtour elimination constraints required to reach optimality. We can also evaluate whether the beneﬁts of this solution approach depend on the relationship between the desired number of clusters, p, and the number of clusters generated by the data generation process. The issues addressed in this article can open new research avenues for this and other exact optimization problems.

## The exact formulation of the p-regions model— Tree version
In this section, for the sake of completeness, we present the Tree version of the p-regions problem, p-regionsTree, devised by Duque, Church, and Middleton (2011). This MIP model creates trees by introducing links between areas. A link between two areas denotes that the two areas are spatially connected (i.e., they are neighboring areas). Areas belonging to a given tree form a region. The main condition for designing regions with trees is that no tree can have cycles. The p-regionsTree prevents cycles via an adaption of Miller, Tucker, and Zemlin (1960) tour-breaking constraints (or MTZ constraints) of the TSP.

In the interest of ensuring readability, we present in this article a complete formulation of the p-regionsTree as presented by Duque, Church, and Middleton (2011).

Parameters:
i; I5 Index and set of areas ; I5 1; /C1/C1/C1 ; nfg ;
cij5
( 1; if areas i and j share a border ; with i; j 2 I and i 6¼ j
0; otherwise;
Ni5 jjcij51
/C8/C9
; the set of areas that are adjacent to area i;
dij5 dissimilarity relationships between areas i and j; with i; j 2 I and i < j;

Decision variables:
tij5
( 1; if areas i and j belong to the same region
0; otherwise;
xij5
( 1; if the arc or link between adjacent areas i and j is selected for a tree graph ;
0; otherwise;
ui5 order assigned to each area i in a subnetwork or tree

Minimize:
Z5
X
i
X
jjj>i
dijtij: (1)

Subject to:
Xn
i51
X
j2Ni
xij5n2p; (2)
X
j2Ni
xij /C20 1 8i51; /C1/C1/C1 ; n; (3)
tij1tim2tjm /C20 1 8i; j; m51; /C1/C1/C1 ; n where i 6¼ j; m 6¼ j; (4)
tij2tji /C20 0 8i; j51; /C1/C1/C1 ; n; (5)
xij2tij /C20 0 8i51; /C1/C1/C1 ; n; 8j 2 Ni; (6)
ui2uj1ðn2pÞ3xij1ðn2p22Þ3xji /C20 n2p218i51; /C1/C1/C1 ; n; 8j 2 Ni; (7)
1 /C20 ui /C20 n2p 8i51; /C1/C1/C1 ; n; (8)
xij 2 0; 1fg 8i51; /C1/C1/C1 ; n; 8j 2 Ni; (9)
tij 2 0; 1fg 8i; j51; /C1/C1/C1 ; nji < j: (10)

The objective function (1) minimizes the sum of pairwise dissimilarities between areas belonging to the same region. Constraint (2) holds that the sum of all selected links (i.e., xij 5 1) equals n–p, which is the number of links required to form p trees. Constraints (3) holds that an area i can have at most one link xij leaving areai. Constraints (4) generates triangulations between all areas belonging to the same region; that is, if areas i, j and m belong to the same region, then there must exist links of the type ti;j between those areas. Constraints (5) guarantees symmetry in the matrixti;j,s o the objective function can be calculated with elements above the main diagonal. Constraints (6) holds that a link between areas i and j in the tree can be selected only if they assign to the same region, that is, tij 5 1. Constraints (7) and (8) are tour-breaking MTZ constraints adapted from Miller, Tucker, and Zemlin (1960). These constraints assign an orderui to each area such that a link of type xi;j must depart from an area i with an order ui that is less than the order assigned to the arrival area uj. Altogether this prevents a cycle of linksxij in the tree of a given region. Finally, Constraints (9) and (10) guarantee variable integrity. In total, the number of constraints in expression (7) is given by P
i2I Ni, whereas the number of constraints in expression (8) is n. The model without these MTZ constraints is hereafter referred to as the relaxed model.

## The subtour elimination constraints for the p-regions model
The subtour elimination approach for the p-regions model, proposed by Duque, Church, and Middleton (2011), consists of an iterative process that begins by solving the relaxed version of the p-regions problem that removes the adapted MTZ constraints from its complete formulation (expression (7) and (8)). Then, subtour elimination constraints are incorporated iteratively, that is, the DFJ constraints of the type:
X
i
X
j
xij /C20j Cj21 8i 2 C; 8j 2 Ni \ C; (11)
where C is the set of areas involved in the cycle, and jCj is the cardinality of C.

After the addition of the DFJ constraints, the model is solved again. This process is repeated until a solution is found that contains no cycles. The ﬁrst solution with no subtours is the optimal solution of the complete p-regions problem. The steps above are systematized in Pseudocode 1 and are referred to as the DFJ- p-regions algorithm in the rest of this article.

In the p-regionsTree, a region can be designed with different trees without affecting the objective function. The region composed of six areas can be obtained with different trees, and the objective function will be the same in all cases. This is because the objective function takes each of the pairwise distances, dij between the areas assigned to the same region (via the decision variable tij). Thus, the pairwise distances included in the objective function will be the same regardless of how the areas within the region are linked together.

This particularity of the p-regionsTree model implies that when a region is infeasible (i.e., discontinuous) because of the presence of a subtour in the tree, the addition of the corresponding DFJ constraint, as presented in Constraints (11), may lead to the same infeasible solution. This is because the same infeasible region can remain intact merely by moving the subtour within the region. This characteristic will require several iterations in the algorithm DFJ-p-regions before solving this infeasibility.

In this article, we propose a minor modiﬁcation of the DFJ constraints adapted by Duque, Church, and Middleton (2011) that solves the situation presented above. It consists of introducing a modiﬁcation in the deﬁnition of the set C in the DFJ constraints (11). Our new deﬁnition of C is the following: a set of areas that belong to the subregion that contains the subtour. In the example described above, the set C would be {7, 8, 9, 10, 11, 12}, and the DFJ constraints for the case illustrated in 2a would be: x
7;81x7;101x8;71x8;91x8;111x9;81x9;121x10;71
x10;111x11;81x11;101x11;121x12;91x12;11 /C20 5. This constraint will also prevent any other subtour within the subregion with areas 7, 8, 9, 10, 11, and 12. Therefore, this speciﬁc infeasible solution will not appear again. In contrast, based on the deﬁnition of the set C proposed by Duque, Church, and Middleton (2011), C and therefore the DFJ constraints would be different for each case: for one subtour, C 5 {7, 8, 10, 11} and the DFJ constraint is x7;81x7;101x8;71x8;111x10;71x10;111x11;81x11;10 /C20 3; for another, C 5 {7, 8, 9, 10, 11, 12} and the DFJ constraint is x7;81x7;101x8;71x8;91x8;111x9;81x9;121x10;71x10;111x11;81x11;101 x11;121x12;91x12;11 /C20 5; for another, C 5 {8, 9, 11, 12} and the DFJ constraint is x8;91x8;111x9;81x9;121x11;81x11;121x12;91x12;11 /C20 3; for another, C 5 {7, 10} and the DFJ constraint is x7;101x10;7 /C20 1. In this article we will use the new deﬁnition of the set C when running the DFJ- p-regions algorithm.

Algorithm. DFJ-p-REGIONS()
Comment: Reach feasibility by adding DFJ constraints.
while true
do
solution 5 solve the relaxed model :
cycles5check for cycles in solution and add them to cycles:
if cycles 6¼ 1
then
(
add subtour elimination constraints to the model based on
Constraints ð11Þ:
else STOP :f
8
>>>
>>>>>
><
>>>
>>>>>
>:
return solution

## Example
To illustrate how the DFJ- p-regions algorithm works, we present a simple example. A regular lattice with nine areas and a grid grayscale coded according to variable y that is used to calculate dissimilarities between areas (i.e., dij 5jyi – yjj).

As a benchmark, an optimal solution of the complete problem is shown. Decision variables u are explicitly shown in the corresponding order, whereas decision variables t and x are used to depict the different regions via links and the grayscale code.

The solution and the subtour elimination constraints added during the execution of the DFJ- p-regions algorithm (with our new deﬁnition of the set C) are presented. The doted arrows denote the presence of a cycle in a region. The two subtours in step 1 are good examples of the new deﬁnition of the set C; on the one hand, the subtour between areas 1 and 2 occurs within a small subregion composed of areas 1 and 2. In this case, C 5 {1, 2} and the DFJ constraint is x1;21x2;1 /C20 1. On the other hand, the subtour between areas 5 and 8 occurs within a larger subregion composed of areas 4, 5, 7, 8 and 9. In this case, C 5 {4, 5, 7, 8, 9} and the DFJ constraint is x4;51x4;71x5;41x5;81x7;41x7;81x8;51x8;71x8;91x9;8 /C20 4. This constraint will prohibit the current subtour and any other subtour that leads to the same subregion. Similar situations occur for the subtour between areas 1 and 4 in subregion {1, 2, 4} in step 2 and for the subtour between areas 5 and 8 in subregion {5, 8, 9} in step 3. When no cycle is present and the minimum gap criterion is achieved, we have found the optimal solution via the DFJ- p-regions algorithm. The optimal solution includes ten DFJ constraints.

## Computational experiment
In this section, we test the performance of the DFJ- p-regions algorithm and compare it with the classical solution approach, which involves solving the complete formulation of the p-regionsTree model. A data set was used in the experiments. We created ﬁve regular lattices of different sizes, n. The attributes, from which dissimilarities dij were determined, were simulated as spatial autoregressive processes with six different values for the spatial autocorrelation parameter q, mean 50, and for spatial weights based on the rook criterion. We also deﬁned three values for the parameter p associated with the number of regions. Finally, for each combination of ½n; p; q/C138, we generated 30 instances. This gives us a total of 2,700 instances solved using both methods.

We implemented both methods in Python 2.7.9 and used Gurobi 5.6.0 to solve the MIPs problems. The connection between Gurobi and Python was carried out using the Gurobi Python Interface: Python was used to identify subtours, and Gurobi was used as an LP solver. For the DFJ- p-regions algorithm, once a MIP incumbent was found, we determined whether cycles were present, and the corresponding subtour elimination constraints were added using Gurobi lazy constraints. The solver was allowed to run for a maximum of 3 h. We ran the experiments on a Dell Power Edge 1950 GIII 8 cores, 2.33GHz Intel Xeon running Linux Rocks 6.1 at 64 bits of the Apolo Scientific Computing Center of Universidad EAFIT.

To identify if a subregion contains subtours, we proceeded as follows. We used the xij to construct directed graphs associated with each subregion. Then, recalling from graph theory, we checked for strongly connected components (SCC) in the directed graphs. A SCC is a maximal induced subgraph where for every pair of nodes there is a path back and forth, that is, a SCC is a subtour in the tree. When a graph has fewer SCCs than nodes, it means that it has subtours. In that case, we added the associated subtour elimination constraint, expression (11), where the members of the set C are the areas of the corresponding graph. We used the NetworkX Python package (Hagberg, Schult, and Swart 2008) to identify the SCC using Tarjan’s algorithm (Tarjan 1972) with Nuutila’s modiﬁcations (Nuutila and Soisalon-Soininen 1994).

Table 2 reports, for each method, the number of instances that converged and those that did not. After 3 h, the DFJ- p-regions algorithm was able to solve 51.41% of the 2,700 instances, whereas the complete formulation solved 41.41% of the instances. This denotes an improvement of 10% in the convergence rate and signiﬁes an important achievement within the context of NP-hard problems.

To compare the results in terms of performance, the following indexes were calculated:

The Speedup is the relationship between the execution time required for the complete formulation, CPUtimeC, and the execution time required for the DFJ- p-regions algorithm, CPUtimeDFJ. Values greater than one denote that the DFJ- p-regions algorithm outperforms the complete formulation.

Speedup5 CPUtimeC
CPUtimeDFJ

The best integer solution (BIS) is the best objective found by the end of the execution time, and the best bound (BB) is the best solution to the relaxed model found through a branch-and-bound search by the end of the execution time. The differences between the BIS and BB is calculated as percentages as follows:

%DBIS5 ðBIST 2BISDFJ Þ
BIST
3100%

%DBB5 ðBBDFJ 2BBT Þ
BBDFJ
3100%

A positive value for % DBIS denotes that the DFJ- p-regions algorithm has found a better feasible solution to the problem at hand relative to the complete formulation. Similarly, a positive %DBB denotes that the DFJ- p-regions algorithm has found a better (i.e., higher) lower bound, implying that it is closer to converging than the complete formulation.

According to results, there are three outcomes: both methods have converged (1,118 instances), only the DFJ- p-regions algorithm has converged (270 instances) and none of the methods have converged (1,312 instances).

### Both methods have converged
For instances where both methods have converged, as expected, the higher the number of areas is, n, the lower the number of instances solved optimally is. The impact of parameter p is also as expected: the larger the p is, the smaller the average size of the clusters is, thus implying less complicated trees and fewer subtour possibilities.

Because both methods found the optimal solution, we focus on comparing execution times. The geometric mean of Speedup is calculated among those instances that fall into this category. The results show that the DFJ- p-regions algorithm overcomes the complete formulation when the ratio n/p is greater than 3. With respect to the spatial dependence parameter q, no particular pattern was identified.

### Only the DFJ- p-regions algorithm converged
For instances where only the DFJ- p-regions algorithm converged, our interest is to determine how far the complete formulation was from the optimal solution. Average % DBIS and % DBB were calculated differentiated by n, p, and q. Values close to zero in % DBB indicate that the complete formulation found an integer solution that is very close to the optimal, but % DBIS shows that it is not close enough to satisfy the optimality criterion (values shown fall between 5.45% and 87.08%). None of the tables shows a clear effect of parameter q.

It is important to note that, to the best of our knowledge, this is the ﬁrst time that an instance of the p-regions problem with 49 areas has been solved to optimality in less than 3 h using exact optimization techniques.

### No method converged
The instances where no method converged show that spatial clustering remains a problem that is very difﬁcult to solve optimally. Neither the complete formulation nor the DFJ- p-regions algorithm were able to solve a single instance with n 5 64. In general, it seems that the higher the value of q, the easier the problem is to solve.

The average % DBIS when no method converged indicates a predominance of negative values, meaning that, after 3 h, the complete formulation found a better integer solution than the DFJ-p-regions algorithm in many cases. For a ﬁxed number of areas, n, the difference between the two approaches tends to increase, in favor of the complete formulation, as p decreases.

## Disentangling the relationship between the DFJ- p-regions and rho
The computational experiments did not show convincing evidence in favor of the direct relationship between the level of spatial autocorrelation and the beneﬁts derived from the DFJ- p-regions approach. At this point, we know that the beneﬁts of the DFJ- p-regions algorithm appear when n=p /C21 3, but the results did not show that spatial patterns with high spatial autocorrelation require a much lower number of subtour breaking constraints. These results appear counterintuitive in the light of our initial thoughts: in the presence of high spatial autocorrelation, the spatially constrained clusters will tend to appear naturally, and the relaxed p-regions will require a few number of subtour breaking constraints to reach optimality. We present in this section a last computational experiment with which we want to provide evidence in support of the following hypothesis: The maximal beneﬁts from the DFJ- p-regions algorithm are obtained when the parameter p matches the number of natural clusters in the spatial pattern of the aggregation variable, p*, that is, when p 5 p*.

For this second experiment, a set of 30 regular lattices of size n 5 36 were generated, each with a spatial pattern designed in such a way that each instance contains p* 5 7 natural clusters. The areas within each cluster have a random value from an uniform distribution. The minimum and maximum limits of the uniform distribution change from one cluster to another so that each cluster contains values that do not overlap with values in other cluster. An example of the spatial distribution obtained with this strategy is presented. The seven clusters can be easily identiﬁed in the spatial pattern and the values within each cluster present small variations.

Our experiment consists of solving each instance using both approaches, the complete p-regions formulation and the DFJ- p-regions algorithm. In both cases we used three different values for p: {3, 7, 11}. Then, using the instances that where solved to optimality, we measured the execution times and the number of subtour elimination constraints required by the DFJ- p-region algorithm to terminate. We compared these results with the results from the experiment described earlier with n 5 36, p: {3, 7, 11}, and q 5 0.

Even though the p-regions formulation is positively affected by the presence of natural clusters, the DFJ- p-regions algorithm signiﬁcantly outperforms its counterpart both in terms of number of instances solved to optimality and computational cost, in particular when p 5 p* . We also found that when p* 5 p 5 7, the variance of the execution times is much lower in the DFJ- p-regions algorithm (with execution times between 6.82 and 16.25 s) than in the complete p-regions formulation (with execution times between 62.07 and 6,694.98 s).

Finally, the number of subtour elimination constraints required by the DFJ- p-regions algorithm shows two different effects: The ﬁrst effect is related to the low number of subtour elimination constraints as a consequence of a low ratio n/p. When p 5 11 the ratio n/p is 3.3, which implies small regions and therefore fewer possibilities for subtours. In this case, both the random and the clustered patterns are beneﬁted. The second effect conﬁrms the hypothesis formulated at the beginning of this section; when p* 5 p 5 7 the clustered pattern requires much less subtour elimination constraints than its counterpart. This last experiment lead us to the following conclusion: the greatest beneﬁts of using the DFJ- p-regions algorithm emerge when (1) the instance exhibits a natural clustered pattern, and (2) the value of the parameter p is close to the number of natural clusters in the instance, p*.

## Conclusions
In this article, we implemented an extensive computational experiment to test the performance of the subtour elimination constraints proposed by Duque, Church, and Middleton (2011) for the p-regions problem (the DFJ- p-regions algorithm). We placed special attention to the impact of the levels of spatial autocorrelation of the aggregation variable on computational time.

The results show that the higher the spatial dependence, the lower the number of subtour breaking constraints required to optimally solve the problem and, thus, results in shorter execution time. This is because clustered spatial patterns tend to create spatially contiguous regions. We found that the greatest beneﬁts are obtained in two situations: (1) when n=p /C21 3; and (2) when the parameter p is close to the number of clusters in the spatial pattern of the aggregation variable (p*).

In this article we increased the convergence rate by roughly 12%, allowing us to optimally solve, with exact optimization, the largest instance of the p-region-problem from its ﬁrst appearance in Duque, Church, and Middleton (2011). This constitutes an important step forward in the simpliﬁcation of this NP-hard problem. In the future, we plan to explore other decomposition techniques that allow us to optimally solve larger instances of the problem (e.g., Lagrangian relaxation, Benders decomposition, and the so-called Dantzig–Wolfe decomposition, which is also known as column generation).