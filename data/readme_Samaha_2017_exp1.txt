Contributor: Jason Samaha

Contact: jsamaha@ucsc.edu

Citation:  Data from Experiment 1 of Samaha, J., & Postle, B. R. (2017). Correlated individual differences suggest a common mechanism underlying metacognition in visual perception and visual short-term memory. Proc. R. Soc. B, 284(1867), 20172035. https://doi.org/10.1098/rspb.2017.2035 


Stimulus: Stimuli were sinusoidal luminance gratings presented in a circular aperture and averaged with white noise (see Fig 1). There were two different tasks in the experiment. 1) Orientation estimation (Task == “percept”), where a single noisy grating with random orientation was presented and, 600 msec later, subjects saw a new grating that they could rotate with the computer mouse to match the orientation of the target grating. Subjects then rated confidence on a 1-4 scale. 2) Delayed orientation estimation (Task == "wm"), followed the same procedure as the perceptual task, but with a 7 second delay between target and response. "Stimulus" is the orientation (in degrees) of the target and varied between 1-180 degrees, randomly selected on each trial. "Response" is coded between 1-360 degrees, so to compute error, one needs to select the minimum circular distance between Stimulus and Response. This is what is computed in the "Error" variable - negative is counterclockwise error.

Confidence Scale: the scale ranges from 1 - 4, with one responses explained to the subject as “guessing” and 4 responses explained as “highly confident”.

Manipulations: Half the stimuli (randomly selected) had "High positive evidence" or 
"Low positive evidence" ("Condition" == 1 or 0.5, respectively). Low PE stimuli were derived from high PE stimuli by halving the Michelson contrast of the grating and the noise prior to averaging the two to make the final stimulus (see methods of original paper for more details). 

Block size: The perceptual task comprised two blocks of 120 trials each - one at the beginning and one at the end of the experiment. The working memory task comprised 3 block of 60 trials each. 

Feedback:  no feedback was used.

Training: Prior to the main task, a staircase was used to achieve performance around 78% accuracy. The staircase data is not included here.

Link to original analysis: https://osf.io/ahm6c/ 

Notes: a) due to a programming error, the orientations were not selected completely randomly for the first 14 subjects, but had a bias towards orientations between 0 and 90. Plot histogram(d.Stimulus) to see the bias. b) subject 10 aborted the first WM block early, and did not finish the last perceptual block, leaving 292, rather than 420 total trials for this subject.