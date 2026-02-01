# Delineating school attendance boundaries to improve diversity and community integrity

## Abstract
The delineation of school attendance zones (SAZs) is a critical aspect of educational policy, influencing student distribution, resource allocation, and educational equity. SAZs determine which students attend specific schools based on geographic boundaries, shaping the demographic composition of schools and the opportunities available to students. Designing SAZs is a complex and challenging task, as it requires balancing multiple, often conflicting objectives and constraints. These include minimizing student travel time, promoting diversity and reducing segregation, balancing school capacities to prevent overcrowding or underutilization, and maintaining zone contiguity. In this paper, we present a multi-objective spatial optimization model to address these competing priorities. We applied this method to redraw the attendance boundaries of elementary schools in the Riverside Unified School District, California. The results demonstrate that our model can significantly reduce school segregation while minimizing student travel time and maintaining contiguous attendance zones. By generating a set of Pareto-optimal solutions, this approach equips policymakers with the tools to make informed decisions on SAZ delineation that improve both equity and logistical efficiency.

Keywords
school districting, segregation, contiguity, optimization, minimizing student travel time

## Introduction
Federal policies on school districting in the United States are designed to promote equitable educational opportunities for all students, but actual implementation often varies widely at the state and local levels. The Every Student Succeeds Act (ESSA), signed into law in 2015, provides protections for disadvantaged and high-need students and mandates high academic standards for all. Additionally, the Biden administration’s Equity Action Plan furthers this mission by addressing disparities and supporting underserved communities. While the federal government establishes broad guidelines and provides funding, the responsibility for districting and resource allocation ultimately lies with states and local school districts.

State-level policies that have attained a certain level of agreement prioritize some important aspects including fair distribution of resources, expanding school choice options, ensuring school safety, and promoting community engagement, among others. Hanson (2024) shows that some states like Massachusetts, Connecticut, and New Jersey are notable for their substantial per-pupil spending on public schools, ensuring quality education for all children regardless of income, disability, or prior academic performance. On the other hand, states like Alabama, Mississippi, and Nevada struggle to provide the same level of opportunity due to lower per-pupil spending and heavy reliance on local property taxes. This funding structure often results in under-resourced schools in lower-income areas, while wealthier districts benefit from better-funded educational facilities.

The impact of these disparities is evident in public school assignment practices across the United States. Many school districts utilize assignment policies largely determined by residential addresses, which results in higher-income families having access to better-resourced schools while lower-income families are relegated to underfunded districts due to significant differences in local property taxes and housing costs ( Clapp et al., 2008 ; Jud and Watts 1981 ; Kane et al., 2006 ). Efforts to address these disparities include the implementation of school choice programs, such as charter schools, magnet schools, and open enrollment policies, which allow families more flexibility in selecting schools outside of their designated districts ( Rivkin and Welch 2006 ). Even with increasing flexibility in school choice, more than 70% of children still attend the assigned public school based on school attendance zone (SAZ) boundaries delineated by school districts ( De Brey et al., 2021 ). To further promote equity and inclusivity, the U.S. Supreme Court strongly encouraged, and many states have undertaken reforms to adjust boundaries of SAZs to account for ethnic and social diversity ( Carlson et al., 2020; Saporito and Van Riper 2016). Particularly, several school districts have adopted student assignment policies requiring a set proportion of minority students, while others mandate a specific percentage of socioeconomically disadvantaged students (Carlson et al., 2020 ; Reardon and Rhodes 2011 ).

Although efforts have been made to reduce segregation in schools, these actions may inadvertently create new challenges. Parents Involved in Community Schools v. Seattle School District No. 1 (2007) is a landmark U.S. Supreme Court case that addressed the use of race in public school student assignment plans. More specifically, the Seattle School District used a racial factor as a tiebreaker to decide which students would be admitted to oversubscribed high schools. If the demographics of a school significantly deviated from those of the district, the racial tiebreaker would give preference to underrepresented groups. Parents for Engaged Community Schools, a nonprofit organization, launched a lawsuit contesting the constitutionality of this racial tiebreaker plan and the Supreme Court ultimately ruled in their favor with a 5-4 decision. The notable case underscored the limitations of using race as a criterion in public school assignments, emphasizing that any race-based action must be thoroughly scrutinized and justified. Even when taking race into account, it should be treated as one of several aspects and may not be the decisive factor.

In addition to these widely recognized priorities, at the local level, there has often been an implicit or explicit preference for creating contiguous school attendance zones. This preference is frequently tied to the goals of maintaining neighborhood cohesion and simplifying transportation routes (FCPS 2024; Sistrunk et al., 2023), although the empirical evidence supporting these benefits over other considerations, such as diversity and equity, is varied and has been increasingly questioned ( Asson et al., 2024 ; Richards, 2014 ; Saporito and Van Riper, 2016 ).

Delineating SAZs is a complicated task due to the need to balance multiple, often conflicting objectives and constraints. In this paper, we propose a new student assignment approach that can minimize travel time for students and reduce segregation while balancing school capacities to prevent overcrowding or underutilization, and maintaining contiguity of zones. The next section provides a review of various school districting methods, followed by a detailed description of our proposed model. We then applied this model to the Riverside Unified School District in California, USA to demonstrate how to use this new method to assist with SAZ delineation. We close the paper with a discussion of our key findings and suggestions for future research.

## Background
A variety of optimization models with diverse objectives and constraints have been used to determine SAZ boundaries in order to facilitate public school student assignment ( Caro et al., 2004 ; Delmelle et al., 2014; Ferland and Guénette 1990; Heckman and Taylor 1969; Lemberg and Church 2000; Schoepfle and Church, 1991 ; Wei et al., 2022 ). Minimizing travel distance and/or time has been a key objective of several initial endeavors since 1960s (e.g., Franklin and Koenigsberg, 1973; Jennergren and Obel 1980 ; Lemberg and Church 2000 ; Maxfield 1972 ; Schoepfle and Church 1991). Reducing travel cost could lower transportation expenses for school districts, encourage students’ active school travel, decrease the potential dangers of long commutes, and facilitate regular school attendance, particularly for students from low-income families who may lack reliable transportation ( Andersson et al., 2012 ; Li and Zhao 2015 ; Liu et al., 2024 ). Consequently, initial research formulated the process of defining SAZ as a classic transportation or network flow problem, considering the assignment of students to schools as a commodity that flows from neighborhoods to schools ( Belford and Ratliff 1972 ; Jennergren and Obel 1980 ; Maxfield 1972 ; Schoepfle and Church 1991 ). Nevertheless, the network flow approach does not effectively capture many distinctive requirements of school district planning, such as the geographical contiguity of SAZs, the variable capacity of schools, the necessity for racial balance, and many diverse, often conflicting objectives ( Lemberg and Church 2000 ).

Geographical contiguity also plays an important role in school redistricting since contiguous districts are strongly preferred in many school districts based on assumptions that contiguity promotes community integrity and facilitate easier access to schools ( Sistrunk et al., 2023 ). For example, the Fairfax County adjusted school boundaries to eliminate attendance islands, a geographic area that is assigned to a school although the area is not contiguous to the attendance area(FCPS 2024 ). Many studies have used geographic contiguity as a constraint, either implicitly or explicitly, in their optimization models for defining SAZs (Franklin and Koenigsberg 1973; Shirabe 2009). Nevertheless, in contrast to the legal requirements many U.S. states impose on Congressional redistricting—such as mandating geographic contiguity —no such legal standard exists for the contiguity of school attendance boundaries. Recent research has highlighted that forcing contiguity can directly conflict with other important objectives, such as achieving racial or socioeconomic balance, especially in residentially segregated areas. For instance, Saporito (2017) demonstrated that school attendance zones with irregular shapes —including those comprising multiple non-contiguous areas—tend to exhibit lower levels of income segregation. Similarly, recent research by Asson et al. (2024) finds that non-contiguous zoning is more often linked to greater racial diversity within schools, as it enables the deliberate inclusion of demographically distinct neighborhoods.

A significant body of literature has dedicated to examining aggregated patterns and trends among racially diverse public school districts ( Logan et al., 2008 ; Reardon et al., 2000 ; Richards 2014). Various segregation indices, such as the dissimilarity index ( Logan et al., 2008 ; Logan and Oakley 2004), Gini index (Rivkin 1994), and entropy index (Reardon et al., 2000), have been used to measure or assess racial/ethnical segregation in public schools. Several optimization models incorporate diversity requirements into the formulation of school districting problem; however, most just establish thresholds for targeted minority populations at schools rather than explicitly integrating segregation indices into objective functions or constraints (Church and Murray 1993; Lemberg and Church, 2000; Liggett 1973; Wilson 1985). For example, the classic formulation of the Generic School Districting Problem (GdiP) and its extensions specify the minimum/maximum minority enrollment counts or percentages at each school (Lemberg and Church 2000; Schoepfle and Church 1991). Wei et al. (2022) conducted one of the few studies that explicitly use the minimization of segregation indices as districting objectives. They calculated racial segregation using the dissimilarity index among schools in a district and set minimizing the dissimilarity index as the single objective for student assignment. However, their model often led to substantial increases in student travel time and resulted in many non-contiguous SAZs. Even with geographic proximity constraints, their single objective to minimize racial segregation somewhat negatively impacted another important goal of school districting: ensuring accessible educational opportunities for all students.

This leads to a significant research gap. While many school districting models has the objective of minimizing transportation cost and a few has the goal of minimizing segregation, there is a lack of method that integrates both important goals. Although some models enforce contiguity and others ignore it, few provide a mechanism to control the degree of non-contiguity. A model that simply minimizes segregation may produce highly fragmented zones that are logistically or politically infeasible. Therefore, a more comprehensive method is needed —one that can minimize segregation and travel costs while allowing for non-contiguity in a controlled manner.

## Method
In this paper, we first present a multi-objective spatial optimization model aimed at designing SAZs that can minimize student travel distances and reduce racial segregation within a school district. Consider the following notation.

Parameters
i, j = index of census units (I entire set)
k = index of schools (K entire set)
Si = number of students in census unit i
Ni = number of minority students in census unit i
Wi = number of majority students in census unit i
Ai = adjacent neighbors of census unit i
Ck = capacity of school k
Dik = transport cost from census unit i to school k
FChigh k = fractional upper bound on capacity usage of school k
FClow k = fractional lower bound on capacity usage of school k
Tk = maximum transport cost allowed for students assigned to school k

Decision variables
xik =
/C26
1, if students of unit i are assigned to school k
0, otherwise

The number of students in census unit i is denoted as Si, while the number of minority students is represented by Ni and the number of majority students is represented by Wi. Each school has a predetermined capacity, Ck. The transportation cost between census unit i and school k, typically measured in travel distance or travel time, is precomputed and denoted as Dik. The capacity usage of each school is constrained by a fractional lower bound FClow k and upper bound FChigh k. A maximum transport cost is prespecified for each school to ensure that no student needs to travel too far to go to school.

Binary decision variables xik are used to ensure that all students from a single census unit are assigned to the same school, as splitting neighborhoods is generally undesirable due to potential negative social and political impacts ( Caro et al., 2004 ; Lemberg and Church 2000 ). With this notation the multi-objective school districting model is formulated as follows:
min
X
i
X
k
Dik Sixik (1)

min 1
2
X
k
/C12/C12
/C12/C12/C12
/C12
P
i xik Ni
N /C0
P
i xik Wi
W
/C12/C12
/C12/C12/C12
/C12
(2)

Subject to:
X
k
xik ¼ 1, "i 2 I (3)

X
i
Sixik ≤
/C0
1 þ FChigh
k
/C1
Ck , "k 2 K (4)

X
i
Sixik ≥
/C0
1 /C0 FClow
k
/C1
Ck , "k 2 K (5)

Dik xik ≤ Tk , "i 2 I, k 2 K (6)

xik 2 f0, 1g, "i 2 I, k 2 K (7)

The objective (1) is to minimize the total travel costs of the students of each census unit to their assigned schools. The objective (2) is to minimize the racial segregation among schools in the school district that is measured using the dissimilarity index. The dissimilarity index is one of the most widely used segregation measure to assess school district racial/ethnic segregation ( Farley 1984; Logan et al., 2008; Logan and Oakley 2004; Wei et al., 2022; Wilson 1985). The dissimilarity index represents the percentage of the students that would have to change their schools for the two social groups to be evenly distributed across the entire school district. Constraint (3) ensures that each census unit is assigned to exactly one school. Constraints (4) and (5) stipulate that no school violates capacity usage lower bound and upper bound. Constraints (6) specify the maximum travel cost allowed for a school assignment. Constraints (7) impose binary conditions on decision variables.

While this model integrates both travel cost and segregation minimization into the objectives, the optimal solutions of this model may contain highly fragmented zones that are logistically or politically infeasible. As mentioned previously, there are some previous models that impose strict constraints that require each SAZ zone is a single, unbroken, and continuous geographic area (Shirabe 2009 ). However, unlike congressional districting, contiguity is a desirable but not mandated feature when delineating SAZs. In fact many SAZs are not one single contiguous zone, and multiple disconnected zones could be assigned to one school in various school districts (Saporito 2017 ). In addition, it may require the compromise of other important goals, such as segregation reduction ( Asson et al., 2024 ; Saporito 2017). To address this issue, we introduce a set of constraints that allows us to control the degree of non-contiguity of SAZ zones. Consider the following additional notation.

Q = number of contiguous zones
M = large number
vik =
/C26
1, if unit i is selected as a sink in the SAZ of school k
0, otherwise

yijk = accumulated flow from unit i to j destined for sinks in SAZ of school k

Q is the number of contiguous zones to be identified. It is important to note that Q may not be the same as the number of schools. In other words, multiple contiguous zones might belong to the same school to allow some degree of non-contiguity. Additional decision variables, including vik and yijk, are introduced to control the non-contiguity of SAZs.

X
j2Ai
yijk /C0
X
j2Ai
yjik ≥ xik /C0 Mvik , "i 2 I, k 2 K (8)

X
j2Ai
yijk ≤ ðM /C0 1Þxik , "i 2 I, k 2 K (9)

X
k
X
i
vik ¼ Q (10)

vik ≤ xik , "i 2 I, k 2 K (11)

vik 2 f0, 1g, "i 2 I, k 2 K (12)

yijk ≥ 0, "i, j 2 Ai, k 2 K

Constraints (8)–(11) are necessary for ensuring the contiguity of Q identified zones. These contiguity constraints are an extension of the network flow contiguity constraints in Shirabe (2005) and Murray et al. (2022) to allow for Q contiguous zones identified and each zone assigned to a school. Specifically, we consider the set of census units as a network where each unit is a node and a bidirectional arc exists between any two adjacent census units. A zone is defined as a sub-network where one node acts as a sink, and all other nodes supply one unit. For the zone to be considered contiguous, the supply from each source node must reach the sink without traversing nodes outside the sub-network. Constrains (8) ensures that for all units assigned to school k, the net out flow for non-sink units is larger than 1 and the net in flow for sink units can be as large as M /C0 1. Constraints (9) limit flow out of a unit unless it is included in a SAZ zone, where the most flow out could be M /C0 1. Shirabe (2005) and Murray et al. (2022) require that each zone only has exactly one sink to guarantee that jKj contiguous zones are identified. However, as mentioned previously, contiguity is a desirable but not mandatory feature for SAZs and many SAZs are not one single contiguous zone. To encourage contiguity, we use constraints (10) to prespecify the total number of contiguous zones (sinks) identified. When Q ¼j Kj, jKj contiguous zones will be identified and each zone is assigned to one school. When Q > jKj, multiple contiguous zones could be assigned to one school. Constraints (11) link unit assignment and sink designation, requiring a unit to be assigned to school k if it is also to serve as the sink for the zone. Constraints (12) impose binary and non-negativity conditions on decision variables.

There are several nuances in this new multi-objective model compared with previous school districting models. First, these two objectives allow us to identify SAZs that can minimize student travel cost while achieving the minimal school segregation. This has not been considered in previous models. Second, the parameter Q combined with constraints (8)-(12) can control how many contiguous zones can be formed. Third, the proposed model establishes a capacity range requirement for each school using FChigh k and FClow k, instead of a simple capacity constraint in many previous models. This approach allows for considering the potential expansion of existing schools and achieving a balanced utilization among all schools. For example, if FChigh k and FClow k are both set at 30%, each school could accommodate between 70% and 190% of its current capacity.

This model has two objectives, meaning that there is no one definitive solution, but rather tradeoff solutions reflecting competing preferences between the transport cost and segregation of each school district. We will use the constraint method ( Cohon, 2013 ), where one objective is integrated into the model as a constraint and the second objective optimized. Specifically, we will convert objective (2) into an additional constraint in the model:

1
2
X
k
/C12/C12
/C12/C12/C12
/C12
P
i xik Ni
N /C0
P
i xik Wi
W
/C12/C12
/C12/C12/C12
/C12
≤ B (13)

where B is a feasible segregation index value. It is important to note that the absolute value can be easily linearized as done in Wei et al. (2022). By varying B across a set of potential segregation goals, different single-objective models result and can be solved using exact mixed integer programming (MIP) approaches. Among these tradeoff solutions we will identify the nondominated or pareto-optimal solutions where no transport cost can be reduced without increasing the segregation value. Each pareto-optimal solution represents a potential SAZ delineation with the minimal transport cost given a segregation threshold.

## Results
We utilized the proposed model to assess racial segregation in the Riverside Unified School District (RUSD) in California, USA. The analysis was based on public school data from the 2015-2016 academic year, sourced from the National Center for Education Statistics (NCES). This data includes school locations, SAZs, and student enrollment numbers for each grade. Additionally, we obtained data on the number of 1st grade students by race at each census block group from the 2014-2018 American Community Survey (ACS) school enrollment data. There are seven racial/ethnic groups: Hispanic, non-Hispanic American Indian/Alaska Native, non-Hispanic Asian, non-Hispanic Black, non-Hispanic White, non-Hispanic Hawaiian Native/Pacific Islander, and non-Hispanic Two or More Races. Here, non-Hispanic White was considered the majority group, while the others were considered minority groups.

There are 30 elementary schools offering 1st grade in RUSD but only 29 of them have delineated SAZs. The 2014-2018 ACS shows that there are 153 block groups within RUSD. These block groups have 3,299 1st grade students, with 623 non-Hispanic White students and 2,676 students of minorities. While many SAZs are contiguous, 10 out of 30 schools are fragmented, such as Victoria Elementary and Jefferson Elementary. You can also find that the current SAZs might split some block groups because the RUSD may not define the SAZs based on census block group boundaries. However, the census block group is the smallest unit for which the Census reports school enrollment by race. We therefore overlaid the block group centroids with the NCES SAZs to determine the current student assignment. We applied the index of dissimilarity using the current student assignment to evaluate the current racial segregation in RUSD. The analysis revealed a dissimilarity index of 0.39 for RUSD, suggesting that 39% of students from either racial group would need to transfer to a different school to achieve an even distribution. According to Logan et al. (2008), a dissimilarity score of 0.39 indicates moderate segregation.

We then applied the proposed multi-objective model to determine if it is possible to identify a SAZ delineation that can improve access and reduce segregation while maintaining the contiguity of most SAZs. Since the NCES did not provide school capacity data, we used the total number of enrolled students as a proxy for each school’s capacity ( Ck ). To simplify, we set both the fractional lower and upper bounds on capacity usage (FClow k and FChigh k ) to 0.3 across all schools. This ensures that the maximum number of students assigned to each school is 1.3 times the current enrollment, and the minimum is 0.7 times the current enrollment. We also compared the current student assignment with school capacity, showing that eight of 29 schools are beyond 130% of the capacity and seven of 29 schools do not reach the 70% of the capacity. We pre-calculated the travel time between block group centroids and schools using ESRI StreetMap Premium, setting the maximum transport cost ( Tk ) to 30 minutes. Since the current SAZs has 10 incontiguous zones, we set the minimum number of contiguous zones ( Q) identified to be 40, which allow at most 10 schools have incontiguous zones. As mentioned earlier, we used the constraint method to identify the pareto-optimal solutions by varying the maximum segregation value from 10% to 40% with a step increase of 1% to ensure that these tradeoff solutions can lead to reduced segregation given the current dissimilarity index of 39.26%.

The two models were implemented in Python and solved using the commercial integer programming solver, Gurobi. The computations were performed on a computer with an Intel Core i9 CPU (2.30 GHz), 32 GB of RAM, running on a Windows operating system. We set a solution time limit of 1 hour when solving these models using Gurobi. The pareto-optimal solutions for model without contiguity constraints were reported in Table 1.

When varying the maximum segregation from 10% to 35% with a step increase of 1%, some models did not find a feasible solution within 1 hour. However, these tradeoff solutions reported are all optimal solutions with an optimality gap of 0.00%. When no contiguity constraints are imposed, we identified 27 pareto-optimal solutions. The minimal total travel time found is 14,393.94 min with an average travel time of 4.36 min per student, and the school district segregation is similar to the current student assignment with a dissimilarity index of 39.39%. The minimal segregation found is 12.97% which is a 67% reduction from the current segregation level, but the travel cost increases to 16,558.61 min with an average travel time of 5.02 min per student. The minimum total travel time for minority students is found when D ¼ 31:99%, and the minimum total travel time for majority students found when D ¼ 37:37%. It is also important to note that the identified SAZs without contiguity constraints are quite fragmented given Q varies from 57 to 70 although only 30 schools exist.

The pareto-optimal solutions for model with contiguity constraints were reported in Table 2. After integrating contiguity constraints, we cannot identify any feasible solutions when D ≤ 0:24. The model also becomes more computationally challenging to solve as the solution time increases quite significantly. With contiguity constraints imposed, we identified 13 pareto-optimal solutions. The minimal total travel time found is also 14,393.94 min with an average travel time of 4.36 min per student, and the school district segregation is similar to the current student assignment with a dissimilarity index of 39.39%. The minimal segregation found is 24.93% which is a 36% reduction from the current segregation level, but the travel cost increases to 14,600.17 min with an average travel time of 4.43 min per student. The minimum total travel time for minority students is found when D ¼ 30:23%, and the minimum total travel time for majority students found when D ¼ 37:37%. All these solutions have identified 39 or 40 contiguous zones, which is similar to the current SAZs.

There is clearly a tradeoff between minimizing total travel time and school segregation. Reducing segregation means that the student travel time would increase, or rather more students would need to travel to further schools. It is also interesting to note that the pareto-optimal solutions for model without and with contiguity constraints have the same objective values when D >0 :24, even though the solutions have completely different number of contiguous zones identified.

When the segregation goal is set as D ≤ 25%, both models (without and with contiguity constraints) yield the same optimal objective values, a total travel time of 14600.17 and a dissimilarity index of 24.93%, which is a 36.50% decrease from the current 39.26%. This is also the minimal segregation that can be reached at RUSD while satisfying contiguity constraints. While the objective values stay the same with and without contiguity constraints, the contiguous zones vary quite significantly. Without contiguity constraints only 12 schools have a single contiguous SAZ zone assigned, whereas the other schools have 2 to 6 contiguous zones assigned. After imposing contiguity constraints 22 schools a single contiguous SAZ zone assigned, seven schools have two contiguous block group zones assigned and one school (Jackson Elementary School) has three contiguous zones assigned.

When segregation goal is set as D ≤ 35%, the segregation increases 40% but the total travel time only decreases 1% to 14,435.37 min compared with the scenario of D ≤ 25%. The objective values stay the same with and without contiguity constraints. Without contiguity constraints only 13 schools have a single contiguous SAZ zone assigned, whereas the other schools have 2 to 6 contiguous zones assigned. After imposing contiguity constraints 22 schools a single contiguous SAZ zone assigned, seven schools have two contiguous block group zones assigned and one school (Hawthorne Elementary School) has three contiguous zones assigned.

## Discussion and conclusion
This study presents a multi-objective spatial optimization model for designing SAZs that addresses the key challenges of minimizing student travel time, reducing racial segregation, and managing geographical zone structure. The application of the model to the RUSD demonstrates its effectiveness in generating a range of policy options and offers valuable insights into school district planning.

The results reveal a clear tradeoff between travel time and segregation reduction. A key insight for planners is that this tradeoff is not linear. As shown in the Pareto curve, initial reductions in segregation can be achieved with modest increases in travel time, but achieving the lowest levels of segregation comes at a much higher marginal cost. This quantitative tradeoff map allows policymakers to make data-driven decisions about how to balance equity goals with logistical and financial constraints.

The model’s ability to control the level of zone fragmentation is another significant contribution. The RUSD results show that substantial desegregation can be achieved while only requiring a small number of non-contiguous SAZ zones. This demonstrates that it is not an all-or-nothing choice between perfect contiguity and extreme fragmentation. A valuable insight is that the model itself, through its travel time minimization objective, tends to find the most geographically compact solutions possible for a given segregation level. This means planners can allow for non-contiguity to break patterns of residential segregation but can trust the model to do so in a way that is as efficient as possible.

While the model provides technically optimal solutions, its implementation faces significant social and political challenges. The very goal of “reducing segregation” is complex. Policies that disperse minority students from their home communities to achieve a district-wide demographic balance can have social costs (Mawene and Bal. 2020). This is analogous to debates over “majority-minority” political districts, where concentrating a minority group can empower them to elect a candidate of their choice, whereas dispersal may lead to greater influence in more districts but less direct representation. Similarly, creating integrated schools may mean that minority students are always in the minority, which could impact their sense of belonging or educational experience (Burdick–Will, 2018). Our analysis showing that minority students may bear a slightly larger travel burden highlights the need to examine the distribution of costs and benefits of any plan.

Furthermore, the public acceptability of non-contiguous zones is a practical hurdle. While this research and other studies show their effectiveness in promoting integration, communities may resist such plans, which can seem unintuitive or unfair to residents who are “skipped over” to be included in a distant school’s SAZ.

Therefore, the model should not be seen as a final arbiter, but as a decision-support tool. The Pareto-optimal solutions it generates should serve as the starting point for a public engagement process. The question of “who decides which solution is optimal?” must be answered by the community itself. School districts can use the maps and tradeoff data generated by the model to facilitate an informed discussion with parents, teachers, and other stakeholders, allowing them to collectively choose a plan that aligns with their shared values and priorities.

This study has several limitations that open avenues for future research. First, we used the dissimilarity index for two groups. Future work should employ multigroup segregation measures, such as the entropy index, to better capture the nuances of diversity in a multiracial context. Second, refining the optimization algorithms to reduce computation times would facilitate the use of the model in larger or more complex districts. Moreover, while our model controls the number of non-contiguous polygons, future models could incorporate objectives to explicitly minimize fragmentation or maximize compactness, providing even greater control over the zones’ geographical character. Finally, exploring the integration of stakeholder preferences in the optimization process could help ensure that the resulting SAZ designs align with community values and priorities.

In conclusion, the proposed spatial optimization model represents a valuable contribution to the field of equitable school district planning. By quantifying the tradeoffs between travel costs and racial integration and by offering a way to manage zone contiguity, it provides a practical tool to help policymakers and communities navigate the complex challenge of designing SAZs that promote educational equity.