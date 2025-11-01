class PolysomnographyAnalysisEngine {
    #sleepStageClassifiers = new EnsembleClassifier();
    #arousalDetectors = new ArousalDetectionPipeline();
    #sleepArchitecture = new SleepArchitectureAnalyzer();
    #respiratoryAnalyzers = new RespiratoryEventDetector();
    
    constructor() {
        this.#initializeSleepStagingModels();
        this.#calibrateArousalDetection();
        this.#establishSleepQualityMetrics();
    }
    
    async analyzeSleepStudy(polysomnographyData) {
        const preprocessedSignals = await this.#preprocessPSGSignals(polysomnographyData);
        const sleepStages = await this.#classifySleepStages(preprocessedSignals);
        const arousalEvents = await this.#detectArousals(preprocessedSignals, sleepStages);
        const respiratoryEvents = await this.#analyzeBreathingPatterns(preprocessedSignals);
        const sleepArchitecture = await this.#computeSleepArchitecture(sleepStages);
        
        return {
            sleepStages: this.#temporalStaging(sleepStages),
            sleepMetrics: await this.#computeSleepQualityMetrics(sleepArchitecture),
            events: {
                arousals: arousalEvents,
                respiratory: respiratoryEvents,
                periodicLimbMovements: await this.#detectPLM(preprocessedSignals)
            },
            sleepArchitecture,
            clinicalInterpretation: await this.#generateClinicalSummary(sleepStages, arousalEvents)
        };
    }
    
    async #classifySleepStages(psgSignals) {
        const epochDuration = 30; // seconds
        const totalEpochs = Math.floor(psgSignals.duration / epochDuration);
        const stagePredictions = [];
        
        for (let epoch = 0; epoch < totalEpochs; epoch++) {
            const epochSignals = this.#extractEpochData(psgSignals, epoch, epochDuration);
            const features = await this.#extractSleepFeatures(epochSignals);
            const stageProbability = await this.#sleepStageClassifiers.predict(features);
            
            stagePredictions.push({
                epoch,
                startTime: epoch * epochDuration,
                stage: this.#selectSleepStage(stageProbability),
                confidence: Math.max(...Object.values(stageProbability)),
                probabilityDistribution: stageProbability
            });
        }
        
        return await this.#applyTemporalSmoothing(stagePredictions);
    }
    
    async #extractSleepFeatures(epochSignals) {
        const featureExtractors = {
            eeg: new EEGFeatureExtractor(),
            eog: new EOGFeatureExtractor(),
            emg: new EMGFeatureExtractor(),
            ecg: new ECGFeatureExtractor()
        };
        
        const features = {};
        
        for (const [modality, extractor] of featureExtractors) {
            if (epochSignals[modality]) {
                features[modality] = {
                    spectral: await extractor.computeSpectralFeatures(epochSignals[modality]),
                    temporal: await extractor.computeTemporalFeatures(epochSignals[modality]),
                    nonlinear: await extractor.computeNonlinearFeatures(epochSignals[modality])
                };
            }
        }
        
        return this.#normalizeSleepFeatures(features);
    }
    
    async #computeSleepArchitecture(sleepStages) {
        const architecture = {
            totalSleepTime: this.#calculateTotalSleepTime(sleepStages),
            sleepEfficiency: this.#calculateSleepEfficiency(sleepStages),
            sleepLatency: this.#calculateSleepOnsetLatency(sleepStages),
            remLatency: this.#calculateREMLatency(sleepStages),
            stageDistribution: this.#calculateStagePercentages(sleepStages),
            awakenings: this.#countAwakenings(sleepStages),
            stageTransitions: this.#analyzeStageTransitions(sleepStages)
        };
        
        architecture.sleepFragmentationIndex = this.#calculateFragmentationIndex(
            architecture.awakenings,
            architecture.stageTransitions
        );
        
        return architecture;
    }
    
    #calculateSleepEfficiency(sleepStages) {
        const totalRecordingTime = sleepStages[sleepStages.length - 1].startTime + 30; // last epoch end
        const totalSleepTime = this.#calculateTotalSleepTime(sleepStages);
        return (totalSleepTime / totalRecordingTime) * 100;
    }
}

class CircadianRhythmAnalyzer {
    #cosinorAnalyzers = new CosinorRegressionEngine();
    #nonparametricMethods = new NonparametricRhythmAnalysis();
    #entrainmentMetrics = new EntrainmentStrengthCalculator();
    
    async analyzeCircadianPatterns(activityData, lightExposure, timeSeries) {
        const cosinorModel = await this.#fitCosinorModel(activityData, timeSeries);
        const nonparametric = await this.#computeNonparametricMetrics(activityData, timeSeries);
        const entrainment = await this.#assessEntrainmentStrength(activityData, lightExposure);
        
        return {
            cosinorParameters: cosinorModel.parameters,
            nonparametricMetrics: nonparametric,
            entrainmentStrength: entrainment,
            phaseMarkers: await this.#estimatePhaseMarkers(activityData, cosinorModel),
            rhythmStability: await this.#assessRhythmStability(activityData, timeSeries)
        };
    }
    
    async #fitCosinorModel(activityData, timeSeries) {
        const cosinor = this.#cosinorAnalyzers.createModel({
            period: 24, // hours
            components: ['mesor', 'amplitude', 'acrophase']
        });
        
        return await cosinor.fit(activityData, timeSeries, {
            optimizationMethod: 'levenberg-marquardt',
            confidenceLevel: 0.95
        });
    }
    
    async #computeNonparametricMetrics(activityData, timeSeries) {
        const metrics = {};
        
        // Interdaily Stability (IS)
        metrics.interdailyStability = await this.#calculateInterdailyStability(activityData, timeSeries);
        
        // Intradaily Variability (IV)
        metrics.intradailyVariability = await this.#calculateIntradailyVariability(activityData, timeSeries);
        
        // Relative Amplitude (RA)
        metrics.relativeAmplitude = await this.#calculateRelativeAmplitude(activityData);
        
        // L5 and M10 calculations
        const l5m10 = await this.#calculateL5M10(activityData, timeSeries);
        metrics.l5Start = l5m10.l5Start;
        metrics.l5Activity = l5m10.l5Activity;
        metrics.m10Start = l5m10.m10Start;
        metrics.m10Activity = l5m10.m10Activity;
        
        return metrics;
    }
}

export { PolysomnographyAnalysisEngine, CircadianRhythmAnalyzer };