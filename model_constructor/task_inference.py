import os
import mimetypes

# transformers version v4.36.1
task_choices = ["AutoModelForCausalLM",
                "AutoModelForMaskedLM",
                "AutoModelForMaskGeneration",
                "AutoModelForSeq2SeqLM",
                "AutoModelForSequenceClassification",
                "AutoModelForMultipleChoice",
                "AutoModelForNextSentencePrediction",
                "AutoModelForTokenClassification",
                "AutoModelForQuestionAnswering",
                "AutoModelForTextEncoding",
                "AutoModelForDepthEstimation",
                "AutoModelForlmageClassification",
                "AutoModelForVideoClassification",
                "AutoModelForMaskedImageModeling",
                "AutoModelForObjectDetection",
                "AutoModelForlmageSegmentation",
                "AutoModelForImageTolmage",
                "AutoModelForSemanticSegmentation",
                "AutoModelForlnstanceSegmentation",
                "AutoModelForUniversalSegmentation",
                "AutoModelForZeroShotlmageClassification",
                "AutoModelForZeroShotObjectDetection",
                "AutoModelForAudioClassification",
                "AutoModelForAudioFrameClassification",
                "AutoModelForCTC",
                "AutoModelForSpeechSeq2Seq",
                "AutoModelForAudioXVector",
                "AutoModelForTextToSpectrogram",
                "AutoModelForTextToWaveform",
                "AutoModelForTableQuestionAnswering",
                "AutoModelForDocumentQuestionAnswering",
                "AutoModelForVisualQuestionAnswering",
                "AutoModelForVision2Seq",
                ]


def infer_task(file_path):
    # 检测文件类型
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type is None:
        return "Unknown file type"

    # 文本文件
    if mime_type.startswith('text'):
        return ["AutoModelForCausalLM",
                "AutoModelForMaskedLM",
                "AutoModelForMaskGeneration",
                "AutoModelForSeq2SeqLM",
                "AutoModelForSequenceClassification",
                "AutoModelForMultipleChoice",
                "AutoModelForNextSentencePrediction",
                "AutoModelForTokenClassification",
                "AutoModelForQuestionAnswering",
                "AutoModelForTextEncoding"]

    # 图像文件
    elif any(mime_type.startswith(t) for t in ['image']):
        return ["AutoModelForDepthEstimation",
                "AutoModelForlmageClassification",
                "AutoModelForVideoClassification",
                "AutoModelForMaskedImageModeling",
                "AutoModelForObjectDetection",
                "AutoModelForlmageSegmentation",
                "AutoModelForImageTolmage",
                "AutoModelForSemanticSegmentation",
                "AutoModelForlnstanceSegmentation",
                "AutoModelForUniversalSegmentation",
                "AutoModelForZeroShotlmageClassification",
                "AutoModelForZeroShotObjectDetection"]

    # 音频文件
    elif any(mime_type.startswith(t) for t in ['audio']):
        return ["AutoModelForAudioClassification",
                "AutoModelForAudioFrameClassification",
                "AutoModelForCTC",
                "AutoModelForSpeechSeq2Seq",
                "AutoModelForAudioXVector",
                "AutoModelForTextToSpectrogram",
                "AutoModelForTextToWaveform"]

    else:
        return "Unsupported file type"
