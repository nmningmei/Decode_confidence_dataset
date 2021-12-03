*data_HainguerlotGajdosVergnaudDeGardelle_unpub*

***Basic details about the experiment:
- Contributors: Marine Hainguerlot (hainguerlot@ese.eur.nl), Thibault Gajdos (thibault.gajdos@univ-amu.fr), Jean-Christophe Vergnaud (Jean-Christophe.Vergnaud@univ-paris1.fr) and Vincent de Gardelle (vincent.gardelle@gmail.com) 

- Citation: There is currently no associated published paper.

- Stimulus: On each trial, after a 250ms fixation cross, two sets of about 100 dots were simultaneously presented for 700ms, one on the left side and one on the right side of the computer screen (stimulus is coded as 1 if the left set contained more dots and 2 if the right set contained more dots). Participants had to indicate which set contained more dots (subject's response is coded as 1 if responded "left" and 2 if responded "right"). After the response, the inter-trial interval was jittered between 0.5s and 1.5s. 

- Confidence scale: Each response was followed by a confidence rating, in which participants indicated their subjective belief that their response just given was correct, on a 6 steps scale ranging from 50% confident (i.e. guess) to 100% confident, in 6 steps of 10% (confidence answer is coded from 0.5 for 50% confident to 1 for 100% confident).

- Manipulation: There was no manipulation. 

- Block size: 512 trials/ subject. Participants completed 5 blocks of about 100 trials each (number of the trial/block is reported in the column 'Trial_in_block'). Between each block participants had a break of 30 seconds.

- Feedback: There was no trial-by-trial feedback. 

***Additional information about the experiment:
- Subject population: 69 individuals (39 females; mean +/- SD age, 23 +/- 2.5 years) 

- Response device: Participants indicated their response by pressing the corresponding arrow on the keyboard (left or right). Participants indicated their confidence rating by using the numerical keys on the top-left of the keyboard. 

- Experiment setting: The experiment was run in the lab with groups of 15-20 participants.

- Training: Instructions about the use of the confidence scale, examples, and a training phase with feedback (40 trials, not included here) were included to make sure that participants understood the mechanism. 

- Special instructions:
--- About the speed of response: Response times shorter than 200ms or longer than 2200ms (from stimulus onset) were discouraged by presenting a "too fast" or "too slow" message on the screen.
--- About the use of the confidence scale: The confidence rating was incentivized using a probability matching rule (Massoni et al., 2014), which is a variant of the Becker-DeGroot-Marschak rule classically used in experimental economics (Becker et al., 1964). 
The participant was offered an exchange between his response and a lottery ticket with a probability P of success. The number P was randomly determined on each trial (with a uniform distribution between 0 and 1), and compared to the confidence response. 
If P was greater than the confidence, then the participants reward was determined by the lottery. If not, it was determined by the accuracy of the response. The mechanism was presented to participants as a way to maximize their earnings by providing accurate confidence ratings. 

- Experiment dates:The data was collected in 2014.

- Location of data collection: At the Laboratory of Experimental Economics in Paris (LEEP). The experiment was conducted in French. The experiment was run using MATLAB (MathWorks) and Psychtoolbox (Brainard, 1997), on screens (resolution 1024 x 768) viewed at normal distance (about 60 cm). 

- Other important information: 
--- About the experimental procedure: Participants came to two sessions, a 'confidence session' and a 'cueing session', which took place 4 days apart. Both sessions started with a working memory test and an initial calibration phase. The order of the two sessions was counterbalanced across participants.   
The submitted dataset only contains data from the 'confidence session'.
--- About the calibration: Stimulus difficulty was calibrated for each participant at the beginning (of each session). Specifically, one circle contained 100 dots, and the other was adjusted with a 2-down 1-up rule (Levitt, 1971), to obtain 71% of left or right responses, in two interleaved staircases of 150 trials each. 
The step size of the staircases was reduced from 20 to 16, 8, 4 and 2 dots on trials 12, 24, 60 and 80 respectively. After the calibration phase, the parameters of the stimuli (i.e. the difference in the number of the dots between the left and right circle) were kept constant, with one adjusted difference for the left stimulus category and one for the right stimulus category. 
The submitted dataset does not contain these calibration trials. 
--- About the coding of response time of decision below 200ms: Response times below 200ms (which concerns very few observations (0.0481% of trials) have not been properly saved. The variable RT_dec has been recoded= 0.2 for RT below 200ms.