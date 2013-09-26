CHART_DIR = ../charts
TARGET_DIR = /media/sf_P/Dropbox/pymlbook/pmle/ch06

copy: 
	#cp $(CHART_DIR)/Spectrogram_Genres.png $(TARGET_DIR)/1400_09_01.png
	cp $(CHART_DIR)/log_probs.png $(TARGET_DIR)/1400_06_01.png
	cp $(CHART_DIR)/pr_pos_vs_neg_01.png $(TARGET_DIR)/1400_06_02.png
	cp $(CHART_DIR)/pr_sent_vs_rest_01.png $(TARGET_DIR)/1400_06_03.png
	
	convert $(CHART_DIR)/pr_pos_vs_rest_01.png $(CHART_DIR)/pr_neg_vs_rest_01.png +append $(TARGET_DIR)/1400_06_04.png
	
	convert $(CHART_DIR)/pr_pos_vs_rest_02.png $(CHART_DIR)/pr_neg_vs_rest_02.png +append $(TARGET_DIR)/1400_06_06.png


	cp *.py $(TARGET_DIR)/code
	cp Makefile $(TARGET_DIR)/code

01:
	python 01_start.py

02:
	python 02_tuning.py

03:
	python 03_clean.py

04:
	python 04_sent.py

log_probs.png:
	python utils.py

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
	

