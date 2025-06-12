#import "@preview/peace-of-posters:0.5.0" as pop

// non chinese text
// abuse regex bug to set tracking for Chinese text only
#show regex("[^\p{scx:Han}]"): set text(
  // font: ("SimSun"),
  lang: "en",
  // tracking: -2pt,
)

// #let en = text.with(lang: "en")
// #let fr = text.with(lang: "fr")
// #let ja = text.with(lang: "ja")
// #let zhl = text.with(lang: "zh")
// #let HanT = text.with(script: "HanT", tracking: -2pt)

// #show text.where(lang: "en"): set text(fill: red)
// #show text.where(lang: "fr"): set text(fill: blue)
// #show text.where(lang: "ja"): set text(fill: yellow)
// #show text.where(lang: "zh"): set text(fill: green)
// #show text.where(lang: "zh", script: "HanT"): set text(fill: purple)
//

#set page("a0", margin: 0.8cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)
#set par(justify: true)

#set text(tracking: -1pt)
#let fignum_size = 42pt

#pop.title-box(
  "基于傅里叶神经算子的超声CT波速重建",
  authors: "杨锦添¹",
  institutes: "¹武汉大学 弘毅学堂",
  authors-size: 0.8em,
  inst-size: 0.72em,
  image: circle(image("/assets/whu-logo.png"), fill: white, inset: -0.75em, height: 6.1em),
  title-size: 2.34em,
)

#let hba = pop.uni-fr.heading-box-args
#hba.insert("stroke", (paint: gradient.linear(green, red, blue), thickness: 10pt))

// and these are for the body.
#let bba = pop.uni-fr.body-box-args
#bba.insert("inset", 20pt)
#bba.insert("stroke", (paint: gradient.linear(green, red, blue), thickness: 10pt))

#pop.common-box(
  heading-box-args: hba,
  body-box-args: bba,
  heading: "概览",
  body: [
    #figure(
      // caption: "超声CT波速重建方法流程图",
      numbering: none,
    )[
      #image(
        "assets/plot.png",
        width: 100%,
      )
    ]
    #place([#text([*A*], size: 64pt)], dx: 0.01cm, dy: -23.5cm)
  ],
)

#columns(
  2,
  [
    #pop.column-box(heading: "背景")[
      超声层析成像 (UltraSound Computer Tomography, USCT) 是一种重要的医学成像技术。它使用主动的超声波源和阵列接收器来捕捉目标内部的信息，以更低的成本提供和传统X光CT相近的信息，但是却不会像X光那样对人体造成伤害。传统的USCT波速重建方法通常基于全波形反演和射线追踪，这些方法在计算效率或计算精度上存在一定的问题。我们希望设计一种在保证精度的前提下，在这个场景中的达到更优的反演性能的方法。
    ]

    #pop.column-box(heading: "数据与方法", stretch-to-next: true)[
      #figure()[
        #image("assets/data.png", width: 90%)
      ]
      #place([#text([*B*], size: fignum_size)], dx: 2cm, dy: -9.5cm)
      #v(-0.25em)
      本模型的数据集来自北京科学智能研究院和中科院声学所举办的比赛"超声CT成像中的波速重建"。该数据集包含了7200对三频段波场(300kHz, 400kHz, 500kHz)和声速数据。他们使用VICTRE (Virtual Imaging Clinical Trial for Regulatory Evaluation) 工具生成3D的人乳房解刨结构，进而得到声速分布，之后再在模型的2D切面上使用CBS求解器得到波场，工作流如图B所示#cite(<zeng2023neuralbornseriesoperator>)。声速分布到波场的过程可以参考图A中的浅蓝色路径。
      // #linebreak()

      在本问题中，我们已知n组波源${f (x_i)}^n$和其对应的$n × j$ 个测点上的声场${u (x_j)}^n_i$，希望可以得到波速$c(x)$的分布。使用传统方法时本问题可以转化为一个带约束的亥姆霍兹方程最优化问题。但是现在我们希望使用端到端的神经网络来解决这个问题，即直接学习
      #v(-0.2em)
      #h(12.5em)
      ${u (x_j)}^n_i stretch(->, size: #300%)^cal(M) c(x) $
      #place([$(x in Omega, Omega"为关注区域")$], dx:27cm, dy: -2.0em)
      #v(-0.25em)
      为了获得对噪声的鲁棒性以及更好的泛化能力，我们在数据进入神经网络前进行了预处理，包括对波场进行包络提取、高频信息提取以及相位提取，如图A所示。这样，$c(x)$可以表示为
      #v(-0.55em)
      $
      c = cal(M)(|cal(H)(u)|, f_("filter") (u), phi(u))
      $
      #v(-0.25em)
      在提取了特征后，我们使用FNO学习映射$cal(M)$。FNO (Fourier Neural Operator)是神经算子学习的代表性模型，具有网格无关性的优良性质。#cite(<li2021fourierneuraloperatorparametric>)
      其学习两个函数空间之间的泛函映射，可以用来解决偏微分方程问题。在这里，我们使用FNO来学习波速分布与声场之间的关系，进而得到波速分布。
      #v(-0.25em)
      #figure()[
        #image("assets/fno_layer.png", width: 84%)
      ]
      #place([#text([*C*], size: fignum_size)], dx: 2cm, dy: -8.5cm)
      #v(-0.25em)
      Fourier Layer是FNO的核心组件。如图C，它用傅里叶变换将输入的信息转换到频域，之后在频域中通过神经网络R学习具有全局性的信息，最后用逆傅里叶变换得到输出。
      #linebreak()
      我们采用SSIM(Structural Similarity Index)、PSNR(Peak Signal-to-Noise Ratio)和LapLoss作为损失函数。其中SSIM关注局域性的相似性；PSNR起到L2Loss的作用，但是相比L2在接近收敛时梯度更大，训练效果更好；LapLoss关注多尺度信息的重建。
      #linebreak()
      我们使用Adam优化器在单张NVIDIA V100 16G SXM2 GPU上进行训练。最终本模型训练了约130小时达到收敛。

    ]


    #colbreak()

    #pop.column-box(heading: "结果与讨论")[
      #columns(
        2,
        [
          我们使用SSIM和PSNR作为评价指标，对比了我们的模型和基线模型的性能。结果如右表所示。其中基线模型是由前文提到的比赛提供的，同样基于神经算子。
          #colbreak()
          #table(
            columns: (1fr, 1fr, 1fr),
            inset: 0.5cm,
            stroke: (x, y) => if y == 0 {
              (bottom: 0.7pt + black)
            } else {
              // (top: 0.2pt + black)
            },
            table.hline(),
            table.header(
              "模型",
              "SSIM",
              "PSNR",
            ),
            table.hline(),
            [Baseline],[0.8807],[13.61],
            [Our Model],[*0.8991*],[*14.37*],
            table.hline(),
          )
        ],
      )
      // 我们使用SSIM(Structural Similarity Index)和PSNR(Peak Signal-to-Noise Ratio)作为评价指标，对比了我们的模型和基线模型（由前文比赛提供）的性能。
      #v(-0.6em)

      可以看到我们的模型相比基线模型在SSIM和PSNR上都有一定的提升。同时在加噪测试中，我们的模型对噪声也有更好的鲁棒性。
      达到这种精度时，我们的模型仍然可以在单张16G V100 GPU上达到12it/s的推理速度，实现了实时的推理，达到了远超传统全波形反演的性能。#super("②")
      #v(-0.5em)
      #figure()[
        #image("assets/compare.png", width: 80%)
      ]
      #v(-0.6em)
      #place([#text([*D*], size: fignum_size)], dx: 2cm, dy: -10.5cm)

      仔细对比本模型的重建结果与真实值（图D），可以看到我们的模型基本可以正确地重建出波速的分布。在较大体块的重建上达到了不错的精度。但是在高频信息的重建上，我们还没有达到最佳水平。推测可能是由于傅里叶算子本身带有的低通滤波特性导致的。在实验中发现使用带有U-Net的U-FNO模型可以改善高频拟合能力，但是却会导致过拟合问题。
    ]

    #pop.column-box(heading: "展望")[
      - 本模型现在是纯数据驱动的，可以加入物理约束进一步提升模型的可靠性
      - 使用了基于计算机模拟的数据集，没有考虑仪器噪声和其他实际问题，导致容易发生过拟合
      - 对于高频信息的重建效果不佳，可以尝试引入多尺度的方法
      - 引入传感器物理位置的信息，将模型泛化到任意传感器位置和排列
    ]

    #pop.column-box(heading: "附录", stretch-to-next: true)[
      = Acknowledgement
      感谢潘雨迪教授在数据处理思路方面提供的灵感和帮助。感谢武汉大学超算中心提供的GPU计算资源。
      #v(-0.8em)
      = Footnotes
      ① 推测数据集生成使用此公式，具体请参见#cite(<zeng2023neuralbornseriesoperator>)
      #linebreak()  
      ② USCT中的全波形反演往往需要以小时计的时间，而我们的模型可以在毫秒级时间内完成推理#cite(<Fa2018USCTTime>)

      #let bibtext = text.with(size: 0.8em)
      #bibtext(bibliography("bibliography.bib"))
    ]
  ],
)

#let bottext = (a, ..args) => box(text(a, size: 0.65em), ..args)

#pop.bottom-box()[
  #columns(
    3,
    [
      #h(5.5em)
      #box(figure(image("assets/whu-name-w.png", height: 1.5em)), baseline: 1.8em)
      #colbreak()
      #h(3em)
      #box([
        #bottext([Email: #link("mailto:jt_yang@whu.edu.cn")])
        #linebreak()
        #h(-0.22em)
        #bottext([WeChat: yuxizaaa], baseline: -0.2em)
      ])
      #colbreak()
      #box(
        [#box(image("assets/github-mark-white.svg"), width: 8%) #box(
            text(" Daillusorisch/FNO-USCT", size: 42pt),
            baseline: -0.3em,
          )],
        baseline: 1.6em,
      )
    ],
  )

]

