# New Yorker Caption Preprocessing

**Main Article: [Image-to-Text Generation for New Yorker Cartoons](https://eugene-c-tang.medium.com/image-to-text-generation-for-new-yorker-cartoons-f5f145eb6208)**

---

Code used for preprocessing and filtering the New Yorker cartoon captions provided in https://github.com/nextml/caption-contest-data

```
from preprocess_captions import get_file_id_to_captions

caption_starts = {"i", "you", "we", "he", "she", "they", "im"}
file_id_to_captions = get_file_id_to_captions(caption_starts, 15)
print(file_id_to_captions[583])
```
```
['he stole food from the office refrigerator',
 'im guessing a man bun',
 'they dont write people up the way they used to',
 ...]
```
