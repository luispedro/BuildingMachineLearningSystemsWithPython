CHART_DIR = charts

fft:
	python 01_fft_based_classifier.py

ceps:
	python 02_ceps_based_classifier.py

rocs_fft.png: 
	convert $(CHART_DIR)/roc_Log_Reg_FFT_classical.png $(CHART_DIR)/roc_Log_Reg_FFT_jazz.png +append row1.png
	convert $(CHART_DIR)/roc_Log_Reg_FFT_country.png $(CHART_DIR)/roc_Log_Reg_FFT_pop.png +append row2.png
	convert $(CHART_DIR)/roc_Log_Reg_FFT_rock.png $(CHART_DIR)/roc_Log_Reg_FFT_metal.png +append row3.png
	convert row1.png row2.png row3.png -append $(CHART_DIR)/rocs_fft.png
	
rocs_ceps.png: 
	convert $(CHART_DIR)/roc_Log_Reg_CEPS_classical.png $(CHART_DIR)/roc_Log_Reg_CEPS_jazz.png +append row1.png
	convert $(CHART_DIR)/roc_Log_Reg_CEPS_country.png $(CHART_DIR)/roc_Log_Reg_CEPS_pop.png +append row2.png
	convert $(CHART_DIR)/roc_Log_Reg_CEPS_rock.png $(CHART_DIR)/roc_Log_Reg_CEPS_metal.png +append row3.png
	convert row1.png row2.png row3.png -append $(CHART_DIR)/rocs_ceps.png
	
roc_pr.png: fft
	convert $(CHART_DIR)/pr_Log_Reg_FFT_country.png $(CHART_DIR)/roc_Log_Reg_FFT_country.png +append roc_pr.png

sox sine_a.wav sine_b.wav sine_mix.wav:
	sox --null -r 22050 sine_a.wav synth 0.2 sine 400
	sox --null -r 22050 sine_b.wav synth 0.2 sine 3000
	sox --combine mix --volume 1 sine_b.wav --volume 0.5 sine_a.wav sine_mix.wav

fft_demo: sine_a.wav sine_b.wav sine_mix.wav
	python fft.py 
	convert sine_a_wav_fft.png sine_b_wav_fft.png sine_mix_wav_fft.png -append fft_demo.png
	
	python fft.py /media/sf_P/pymlbook-data/09-genre-class/genres/jazz/jazz.00012.wav
	mv jazz.00012_wav_fft.png fft_example.png
	

