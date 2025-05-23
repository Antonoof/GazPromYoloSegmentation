from pathlib import Path
from pipeline import YoloPipeline

pipeline = YoloPipeline(
    model_path="model/best.pt", 
    image_size=(1152, 31920), 
    work_dir="work"  # <-- Лучше не в корень "/"
)

# Приходит запрос с изображением
test_image = Path("11-170-ls-34-g01.png")

# Отправляем в Pipeline
pipeline.process_image(test_image)

# Ссылка на итоговую маску, можно вернуть json(менять get_output_dir)
print("Результаты сохранены в:", pipeline.get_output_dir())
