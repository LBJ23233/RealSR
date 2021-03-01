# :rocket: BasicSR

Based on [BasicSR](https://github.com/xinntao/BasicSR)
---

BasicSR (**Basic** **S**uper **R**estoration) is an open source **image and video restoration** toolbox based on PyTorch, such as super-resolution, denoise, deblurring, JPEG artifacts removal, *etc*.<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://github.com/xinntao/HandyView), [HandyFigure](https://github.com/xinntao/HandyFigure), [HandyCrawler](https://github.com/xinntao/HandyCrawler), [HandyWriting](https://github.com/xinntao/HandyWriting))</sub>


## :turtle: Dataset Preparation

- Please refer to **[DatasetPreparation.md](docs/DatasetPreparation.md)** for more details.
- The descriptions of currently supported datasets (`torch.utils.data.Dataset` classes) are in [Datasets.md](docs/Datasets.md).

## :computer: Train and Test

- **Training and testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.
- **Options/Configs**: Please refer to [Config.md](docs/Config.md).
- **Logging**: Please refer to [Logging.md](docs/Logging.md).

## :european_castle: Model Zoo and Baselines

- The descriptions of currently supported models are in [Models.md](docs/Models.md).
- **Pre-trained models and log examples** are available in **[ModelZoo.md](docs/ModelZoo.md)**.
- We also provide **training curves** in [wandb](https://app.wandb.ai/xintao/basicsr):

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## :memo: Codebase Designs and Conventions

Please see [DesignConvention.md](docs/DesignConvention.md) for the designs and conventions of the BasicSR codebase.<br>
The figure below shows the overall framework. More descriptions for each component: <br>
**[Datasets.md](docs/Datasets.md)**&emsp;|&emsp;**[Models.md](docs/Models.md)**&emsp;|&emsp;**[Config.md](Config.md)**&emsp;|&emsp;**[Logging.md](docs/Logging.md)**

![overall_structure](./assets/overall_structure.png)

## :scroll: License and Acknowledgement

This project is released under the Apache 2.0 license.<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

## :earth_asia: Citations

If BasicSR helps your research or work, please consider citing BasicSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {BasicSR},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2020}
}
```

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR. https://github.com/xinntao/BasicSR, 2020.

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com`.
