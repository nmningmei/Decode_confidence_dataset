Contributor
Mael Lebreton(mael.lebreton@gmail.com)

Citation
Lebreton M, Langdon S, Slieker MJ, Nooitgedacht JS, Goudriaan A, Denys D, Luigjes J**, van Holst R** (2018). Two sides of the same coin: monetary incentives concurrently improve and bias confidence judgments. Science Advances 4 (5), eaaq0668
https://doi.org/10.1126/sciadv.aaq0668
The dataset combines the data from the 4 experiments reported in the publication.
Note that Matlab-formatted data and Matlab analysis codes to reproduce all analyses reported in the paper are available: https://figshare.com/articles/Codes_and_Data/6126776

Stimulus
2 Gratings (gabor patches) of different contrasts embedded in noise, displayed on the left (Stim1) and right (Stim2) side of the screen. The gratings are randomly oriented. Subjects must choose the gabor with the highest contrast.
Stim1 and Stim2 correspond to the value of the contrast for the respective stim
Stmulus indicate the stimulus with the highest contrast (1: left; -1: right), i.e. the correct response.

Response 
1: left; -1: right

Confidence scale
11-point scale: 50% to 100%, with a 5% precision. Confidence accuracy is incentivized with an auction mechanism called probability matching, inspired from Becker-DeGroot-Marshak mechanisms (see publication), such that participants have to report the most precisely and thurthfully confidence as the probability of their choice being correct to maximize their earnings.

Manipulations
Different incentives/rewards. In some contexts, confidence accuracy is rewarded with monetary gains, and in some contexts with the avoidance of monetary losses. 
Incentive conditions are coded as follow (G: gain / L: Loss / N: neutral): 
For experiments 1 and 2: 1: G1€; 2: N; 3: L1€
For experiments 3: 1: G1€; 2: G10c; 3: L1€; 4: L10c
For experiments 4: 1: G2€; 2: G1€; 3: G10c; 4: L2€; 5: L1€; 6: L10c

Block size
Depend on the experiment.
In experiment 1: 4 blocks of 96 trials (subjects coded as 10XX).
In experiment 2: 2 blocks of 54 trials with confidence accuracy incentivization (subjects coded as 21XX), and 2 blocks of 54 trials with performance accuracy incentivization (subjects coded as 22XX). Note that the same subjects performed the two types of task, such that 21XX and 22XX refer to the same subject.
In experiment 3: 2 blocks of 120 trials (subjects coded as 30XX).
In experiment 4: 3 blocks of 108 trials (subjects coded as 40XX).

Feedback
Trial-by-trial feedback depend on the incentivization mechanism; In some trials, a random lottery determine if subjects win (coded as 1) or not (coded as -1) . In other trials, the performance of subjects (correct/incorrect) determine if subjects win (coded as 1) or not (coded as -1). Those two types of feedback - feedback_lottery and feedback_performance - are mutually exclusive (and coded as 0 when absent).

RT
Decision were self-paced.

Subject population
Young adults (19-26 years old).

Response device
Computer keyboard.

Training
A small training version of the task (24 trials). This data was not saved and is not included in the dataset.

Experiment goal
Examine the effect of monetary context on confidence calibration and accuracy. 

Main result
Confidence is increase when participants can win money, and decreased when thay can lose money, despite performance being equal in those two contexts.

Experiment dates
Data were collected between March, 2016 and October, 2016.

Location of data collection
Department of Psychology, University of Amsterdam, Amsterdam, Netherlands.

Language of data collection
English/Dutch.