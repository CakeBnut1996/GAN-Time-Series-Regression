# GAN-Regression
This script is used for numerical values prediction. The example is next two-hour traffic speed prediction based on historical speeds. The model structure is as follows. In this project, LSTM is selected to be the form of both generative and discriminative networks. 
<p align="center">
  <img src="https://user-images.githubusercontent.com/46463367/112266358-15164800-8c31-11eb-82ce-5864632ad946.png"/>
</p>

The scripts include models, preprocessing, training and testing. The evaluation criteria is mean absolute percentage error. If comparing the MAPE by model (GAN, LSTM, XGBoost) and time of day, the results are somethng like the following figure.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46463367/112259952-5ce3a200-8c26-11eb-89b1-66a76af2bd63.png"/>
</p>

The code of LSTM is also included in this repository, whose structure is similar to that of GAN. Can simply be downloaded and run on local machine.
