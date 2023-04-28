import os

import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()

links = [
    "https://live.staticflickr.com/1578/24294187606_89069ac7dd_k_d.jpg",
    "https://live.staticflickr.com/5491/9127573526_2999fafead_k_d.jpg",
    "https://live.staticflickr.com/6161/6175302372_76c4db94d0_k_d.jpg",
    "https://live.staticflickr.com/5601/15309578219_aa39bbfad2_k_d.jpg",
    "https://live.staticflickr.com/2465/3622848494_bad3b7ebe1_k_d.jpg",
    "https://live.staticflickr.com/557/19806156284_3ebb5a4046_k_d.jpg",
    "https://live.staticflickr.com/8403/8668991964_7969e1be9f_k_d.jpg",
    "https://live.staticflickr.com/1924/43556503550_f79978a134_k_d.jpg",
    "https://live.staticflickr.com/3799/20240807568_fcdab6a529_k_d.jpg",
    "https://live.staticflickr.com/7344/9886706776_16f9656162_k_d.jpg",
]
target_class_names = ["person", "bicycle", "car"]


# Get Workspace ID from environment variables
workspace_id = sly.env.workspace_id()
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    print("you should put correct workspaceId value to local.env")
    raise ValueError(f"Workspace with id={workspace_id} not found")

# Create new project and dataset
project_info = api.project.create(workspace_id, "Model predictions", change_name_if_conflict=True)
dataset_info = api.dataset.create(project_info.id, "Week # 1")

print(f"Project has been sucessfully created, id={project_info.id}")


# Initialize `sly.nn.inference.Session`
model_task_id = 32996
session = sly.nn.inference.Session(api, task_id=model_task_id)

# Get model meta
model_meta = session.get_model_meta()

# Create tags
meta_high_confidence = sly.TagMeta("high confidence", sly.TagValueType.NONE)
high_confidence_tag = sly.Tag(meta_high_confidence)

meta_need_validation = sly.TagMeta("need validation", sly.TagValueType.NONE)
need_validation_tag = sly.Tag(meta_need_validation)

# Add tags to project meta
model_meta = model_meta.add_tag_metas(new_tag_metas=[meta_high_confidence, meta_need_validation])
api.project.update_meta(id=project_info.id, meta=model_meta)


for i, link in enumerate(links):
    # upload current image from given link to Supervisely server
    image_info = api.image.upload_link(dataset_info.id, f"image_{i}.jpg", link)
    print(f"Image successfully uploaded, id={image_info.id}")

    # get image inference
    prediction = session.inference_image_url(link)

    # check confidence of predictions and set relevant tags
    image_need_validation = False
    new_labels = []
    for label in prediction.labels:
        if label.obj_class.name not in target_class_names:
            continue
        confidence_tag = label.tags.get("confidence")
        if confidence_tag.value < 0.8:
            new_label = label.add_tag(need_validation_tag)
            image_need_validation = True
            new_labels.append(new_label)
        else:
            new_labels.append(label)

    prediction = prediction.clone(labels=new_labels)
    if image_need_validation is False:
        prediction = prediction.add_tag(high_confidence_tag)
    else:
        prediction = prediction.add_tag(need_validation_tag)

    # upload annotations to Supervisely server
    api.annotation.upload_ann(image_info.id, prediction)
