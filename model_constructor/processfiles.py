from langchain.document_loaders import UnstructuredMarkdownLoader
import os
from collections import Counter


def process_markdown_batch(markdown_files):
    batch_docs = []
    for markdown_file_path in markdown_files:
        markdown_loader = UnstructuredMarkdownLoader(markdown_file_path)
        batch_docs.extend(markdown_loader.load())
    return batch_docs


def iterate_folder_files(root_directory, markdown_files_to_process=[]):
    for root, dirs, files in os.walk(root_directory):
        markdown_files_to_process.extend(
            [os.path.join(root, file) for file in files if file.lower().endswith(".md")]
        )

    return markdown_files_to_process


def process_files_batch(
    process_function,
    markdown_files_to_process=[],
    batch_size=1,
    docs=[],
    processed_files=0,
):
    for i in range(0, len(markdown_files_to_process), batch_size):
        batch = markdown_files_to_process[i : i + batch_size]
        batch_docs = list(map(process_function, [batch]))
        for batch_result in batch_docs:
            docs.extend(batch_result)
            # print(docs)
            processed_files += len(batch)
            # print(f"Processed {processed_files} / {len(markdown_files_to_process)} files")
    return docs


def infer_data_modality(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    text_extensions = [".txt", ".csv", ".json", ".xml", ".md"]
    audio_extensions = [".wav", ".mp3", ".flac", ".aac"]

    _, ext = os.path.splitext(file_path)
    if ext in image_extensions:
        return "image"
    elif ext in text_extensions:
        return "text"
    elif ext in audio_extensions:
        return "audio"
    else:
        return "unknown"


def infer_folder_modality(folder_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    text_extensions = [".txt", ".csv", ".json", ".xml", ".md"]
    audio_extensions = [".wav", ".mp3", ".flac", ".aac"]

    extensions = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            _, ext = os.path.splitext(file)
            extensions.append(ext)

    if extensions:
        most_common_ext, _ = Counter(extensions).most_common(1)[0]
        if most_common_ext in image_extensions:
            return "image"
        elif most_common_ext in text_extensions:
            return "text"
        elif most_common_ext in audio_extensions:
            return "audio"
        else:
            return "unknown"
    else:
        return "empty"


def infer_modality(path):
    if os.path.isfile(path):
        return infer_data_modality(path)
    elif os.path.isdir(path):
        return infer_folder_modality(path)
    else:
        return "invalid"
