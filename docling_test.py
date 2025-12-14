import json
import time
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
from tqdm import tqdm


def main(source):

    input_doc_path = source

    IMAGE_RESOLUTION_SCALE = 2.0


    # Docling Parse with EasyOCR
    # ----------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["es"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_result = doc_converter.convert(input_doc_path)

    ## Export results
    doc_filename = conv_result.input.file.stem
    output_dir = Path(f"./dataset/scratch/{doc_filename}")
    output_dir.mkdir(parents=True, exist_ok=True)

    table_counter = 0
    picture_counter = 0
    for element, _level in conv_result.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                    output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_result.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                    output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_result.document).save(fp, "PNG")

    # Export Deep Search document JSON format:
    # with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
    #     fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    conv_result.document.save_as_markdown(
        output_dir / f"{doc_filename}_with_png.md",
        image_mode=ImageRefMode.EMBEDDED,
    )

if __name__ == "__main__":
    # 遍历./dataset/raw_data/PDF/目录下的所有pdf文件
    pdf_dir = Path("./dataset/raw_data/PDF/")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    need_process_files = []
    for pdf_file in pdf_files:
        output_dir = Path(f"./dataset/scratch/{pdf_file.stem}")
        if not output_dir.exists():
            need_process_files.append(pdf_file)
    for pdf_file in tqdm(need_process_files, desc="处理PDF文件", unit="文件", total=len(need_process_files)):
        output_dir = Path(f"./dataset/scratch/{pdf_file.stem}")
        if output_dir.exists():
            print(f"{pdf_file} 已处理，跳过。")
            continue
        main(str(pdf_file))
