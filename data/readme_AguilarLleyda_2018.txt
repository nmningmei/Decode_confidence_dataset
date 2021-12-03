Readme file for “data_AguilarLleyda_2018.csv”

MAIN INFORMATION:
Contributors: David Aguilar-Lleyda, Maxime Lemarchand, Vincent De Gardelle

Contact e-mail: aguilarlleyda@gmail.com

Citation: Aguilar-Lleyda, D., Lemarchand, M., & de Gardelle, V. (2018). Confidence as a priority signal. bioRxiv, 480350.

Experiment in paper: Experiment 1.

Stimulus: an array of 80 letters (upper-case O or X, colored blue or orange) were shown for one second. They were randomly disposed within an imaginary square, in a very similar fashion as Rahnev et al. (2015, Psych Science).

Response mechanism: after the stimuli disappeared, the response screen was presented, which consisted in 4 square boxes that contained a blue square, an orange square, a white O and a white X. The 4 boxes were randomly arranged in a 2x2 grid on each trial. The task consisted in reporting the dominant letter and the dominant color of the previously seen array. Importantly, the order in which both choices were reported was free. Participants gave their response by clicking on a color box and a letter box, in their preferred order. A first response on one dimension (e.g. clicking on the X box) made the corresponding boxes disappear, leaving only the boxes for the other dimension (e.g. the orange box and the blue box), which required a second response.

Confidence scale: two confidence scales were simultaneously presented, one horizontal for the color task and one vertical for the letter task. Each scale featured the two response categories at its ends (which corresponded to 100% confidence) and an empty region at its center (near which confidence was at 50%). The two choices participants had made on that trial were surrounded by a yellow circle. Participants gave their confidence levels on both tasks by clicking on the corresponding scales, and then validated the ratings by clicking at the center of the screen. Participants were told that maximum uncertainty (no preference between their response and the other option within the same task) corresponded to a 50% confidence, and absolute certainty on the choice to a 100% confidence. 

Manipulations: on each trial, the color and the letter task's difficulty was manipulated independently. Each task could be easy (expected performance of 90% correct) or hard (expected performance of 60% correct). Within the color task, in some trials there were more blue than orange elements, and in others vice versa. The same for the letter task with Os and Xs.

Block size: Participants completed 4 runs of 4 blocks each, with 25 trials per block.

Feedback: after each block, participants were told the percentage of correct choices, pooled across tasks (without differentiating color and letter choices).


DATASET VARIABLES:
Subj_idx: subject identifier
Run: run number
Block: block number (within that run)
Trial: trial number (within that block)
Difficulty_col: expected performance for the color task, in % (90 = easy, 60 = hard)
Difficulty_let: expected performance for the letter task, in % (90 = easy, 60 = hard)
Number_blues: for the color task, number of blue elements, over a total of 80
Number_os: for the letter task, number of Os, over a total of 80
Stimulus_col: dominant dimension for the color task (1 = more blue elements were presented, 2 = more orange elements were presented)
Stimulus_let: dominant dimension for the letter task (1 = more Os were presented, 2 = more Xs were presented)
Response_col: response given for the color task (1 = blue, 2 = orange)
Response_let: response given for the letter task (1 = O, 2 = X)
Accuracy_col: accuracy for the color task (0 = error, 1 = correct response)
Accuracy_let: accuracy for the letter task (0 = error, 1 = correct response)
Confidence_col: confidence rating for the color task (from 50 to 100)
Confidence_let: confidence rating for the letter task (from 50 to 100)
RT_confidence_col: RT for the confidence rating of the color task
RT_confidence_let: RT for the confidence rating of the letter task
Firstchoice: first-reported task (1 = color, 2 = letter)
Location_firstchoice: physical location of the response box that the participant clicked on for her first choice (1 = top-left, 2 = top-right, 3 =bottom-left, 4 = bottom-right)
Location_secondchoice: physical location of the response box that the participant clicked on for her second choice (1 = top-left, 2 = top-right, 3 =bottom-left, 4 = bottom-right)
RT_firstchoice: RT for the first choice
RT_secondchoice: RT for the second choice (the last two variables are coded like this due to the fact that RT for the first choice is always longer, since response physical mapping has to be learnt, and because there are 4 alternatives on the screen, as opposed to only 2 for the 2nd choice)

OTHER INFORMATION:
Subject population: consenting adults, age range approximately 18-50, that participated in exchange for an economic remuneration.

Response device: mouse.

Experiment setting: lab. The room fitted up to 20 people at the same time.

Training: 10 trials where only the color choice (+ confidence) had to be made, 10 trials where only the letter choice (+ confidence) had to be made (these were counterbalanced). Then, participants completed 100 trials of doing both tasks (color and letter) at the same time, where an independent staircase procedure for each task was active.

Experiment goal: see whether participants made first the choice relative to the task they were more confident in (as assessed by the subsequent confidence rating).

Main result: there was a tendency for participants to indeed first respond to the task they were more confident in.

Special instructions: responses were not speeded. Participants were told that they would be rewarded both on the basis of their choices and their confidence ratings (for more details, see methods).

Experiment dates: the first 10 participants were recruited in Spring 2018. The sample was enlarged with 18 more participants in Winter 2019.

Location of data collection: Laboratoire d'Économie Experimentale, Université Paris 1 Panthéon-Sorbonne, Paris, France. The experiment was conducted in French.