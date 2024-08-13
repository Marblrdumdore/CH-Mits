CH-Mits

Due to the limitation on the number of files uploaded per time, we split the whole dataset into 6 parts:
p0-330.zip contains the first 331 POSITIVE samples (ranging from sample 0 to sample 330), and the components of the remaining two datasets (p331-666.zip and p667-1022.zip) can be inferred in the same way (1023 POSITIVE samples in total).
n0-333.zip contains the first 334 NEGATIVE samples (ranging from sample 0 to sample 333), and the components of the remaining two datasets (n334-666.zip and n667-999.zip) can be inferred in the same way (1000 NEGATIVE samples in total).

For both positve and negative datasets, each sample is created by a user from the Chinese social media (i.e., 小红书) and consists of a .png image as well as a .txt file. 

The .txt file has three attributes: note title (笔记标题), note description (笔记描述) and uploading time (上传时间), where the note title stands for the title of the textual content, 
the note description is the main textual content, and the uploading time refers to the time when the note is uploaded.

The dataset is constructed following strict privacy rule, and it does NOT involve any user privacy like user IDs or nick-names.


## Citation

If you find this project useful in your research, please consider cite:

```
@inproceedings{ma2024chmits,
  title={CH-Mits: A Cross-Modal Dataset for User Sentiment Analysis on Chinese Social Media},
  author={Ma, Juhao and Xu, Shuai and Liu, Yilin and Fu, Xiaoming},
  booktitle={33rd ACM International Conference on Information and Knowledge Management},
  year={2024}
}
