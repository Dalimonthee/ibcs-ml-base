# YOLO performance on dashboard chart detection and separation

## Summary

We trained YOLOv8 on synthetic bar charts and ran it on IBCS-style dashboard screenshots. It finds charts often enough to be useful as a first step. It does not reliably produce one box per chart, or keep titles and chrome out of the crop. Validation on synthetic data can look fine. On real dashboards you still see missed charts, merged boxes, and boxes that swallow the whole panel.

## Training setup

The model has one class, `bar_chart`. Training images are procedurally generated: vertical and horizontal bars, grouped and stacked layouts, overlapping series, all at a fixed size. That keeps labels cheap. It also assumes each chart is basically a lone figure on a plain background. Dashboards are not like that. Charts share rows with filters, KPI tiles, legends, and grid lines. They reuse the same fonts and colors as non-chart widgets. Aspect ratios rarely match what the model saw in training.

## Where it falls short

YOLO returns axis-aligned rectangles. Put two charts side by side and you often get one big box, or several small boxes for a single visualization. For a quick pass to see whether anything chart-shaped is present, that is acceptable. For cropping each chart for downstream checks, it is not.

Synthetic variety does not match production UI. Table borders get tagged as charts. Small multiples and thin sparklines get missed.

Full-page screenshots are resized to model input. Small charts become a handful of pixels. Large titles dominate the frame. Confidence scores cluster in the middle, so moving the threshold helps recall or precision, not both.

The actual requirement is closer to layout parsing: panel edges, plot area only, then bar geometry inside. The detector was not trained for that sequence.

## Other approaches

OpenCV-style pipelines can recover bar positions when the scene is simple. They fail on odd chart types too, but they clarify what the neural net skips: structure inside the box. A two-step flow (YOLO proposes regions, classical code measures bars) fits the problem better than treating one bounding box as full compliance.

## Running it on real folders

Predictions on Compliant and non-compliant folders are uneven. Some slides get a confident box on every chart; others get none. Saved overlays are useful for human review. They are not ready as direct input to automated IBCS rules without someone fixing crops first.

## Bottom line

YOLO remains a fast baseline you can retrain when new labels appear. If the goal is stable separation of charts from dashboard chrome and from each other, the current model is mediocre. Improvements probably require fine-tuning on real dashboards, polygon or mask labels, multi-stage cropping, or a layout model. More synthetic epochs alone will not close the gap.
