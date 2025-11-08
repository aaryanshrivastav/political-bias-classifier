from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer & model directly from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("Arstacity/political-bias-classifier")
model = AutoModelForSequenceClassification.from_pretrained("Arstacity/political-bias-classifier")

# Label mapping
label_map = {
    "LABEL_0": "Left",
    "LABEL_1": "Right",
    "LABEL_2": "Center"
}

def classify_long_text(text, chunk_size=512, overlap=50):
    # Initialize classifier
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=chunk_size
    )

    # Split into overlapping chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Run classification on all chunks
    results = classifier(chunks)

    # Aggregate label scores
    label_scores = {}
    for r in results:
        label = r['label']
        score = r['score']
        label_scores[label] = label_scores.get(label, 0) + score

    # Find final label
    final_label = max(label_scores, key=label_scores.get)
    readable_label = label_map.get(final_label, final_label)

    return {
        "final_label": readable_label,
        "label_scores": {label_map.get(k, k): v for k, v in label_scores.items()}
    }


if __name__=="__main__":
    transcript = """
    00:00:00.530 [Music]
    00:00:10.460 in the modern political world the terms
    00:00:13.440 left-wing and right-wing can often be
    00:00:15.690 heard and misunderstood but what do
    00:00:17.400 these terms mean and where do they come
    00:00:19.260 from the terms left and right were first
    00:00:21.539 used in an ideological context during
    00:00:23.460 the French Revolution of the 18th
    00:00:24.900 century those on the left are in support
    00:00:27.180 of the Revolution and those on the right
    00:00:28.830 supported the monarchy this idea of the
    00:00:31.500 left supporting change and the right
    00:00:32.969 wanted to keep the status quo continues
    00:00:35.219 today and is key in some of their
    00:00:36.780 philosophy the difference between left
    00:00:39.600 and right wing ideology fundamentally
    00:00:41.879 comes down to the balance between
    00:00:43.050 individual liberty and government power
    00:00:45.539 the left strive for an equal society and
    00:00:48.539 believed that the state should play a
    00:00:49.860 substantial role in people's lives
    00:00:51.870 this means increased regulation of
    00:00:53.730 business and higher taxes on the rich
    00:00:55.739 the left also tend to hold more
    00:00:57.570 progressive views often opposing the
    00:00:59.579 death penalty while supporting same-sex
    00:01:01.559 marriage and woman's right to abortion
    00:01:03.180 the Left have more lenient views on
    00:01:05.369 immigration and are usually the driving
    00:01:07.380 force behind any separation of church
    00:01:09.090 and state economically the left often
    00:01:12.299 follow the Keynesian or is it sometimes
    00:01:14.250 called the interventionist school of
    00:01:16.049 thought in brief this system would have
    00:01:18.600 the government intervene to avoid an
    00:01:20.460 economic recession this means tax in
    00:01:23.009 highly during Cadore boom times and
    00:01:25.140 spending this money when the economy
    00:01:26.909 truly needs it the level of government
    00:01:29.009 interference varies by how far left it
    00:01:31.439 is with communists wanting complete
    00:01:33.509 control over all aspects of the economy
    00:01:35.400 whereas the center-left
    00:01:37.200 want only moderate intervention the
    00:01:39.119 right believed that a level of social
    00:01:40.710 inequality is inevitable and think that
    00:01:42.869 the government should have a limited
    00:01:44.130 role in people's lives in business this
    00:01:46.140 is as the right believes that preserving
    00:01:48.689 personal freedom should be the
    00:01:50.130 government's main goal and should not
    00:01:51.990 impose too many rules on people's lives
    00:01:54.060 the right also tend to hold more
    00:01:56.189 traditional and religious attitudes than
    00:01:58.020 the left
    00:01:58.439 often opposing things those on the Left
    00:02:00.930 support such as same-sex marriage and
    00:02:03.149 women's right to abortion economically
    00:02:05.219 the right can often be seen using the
    00:02:07.170 new classical approach which includes
    00:02:09.270 have analyzed a fair policy this roughly
    00:02:11.610 translates to leave things alone
    00:02:13.770 and means less regulation to increase
    00:02:15.900 innovation and lower taxes to increase
    00:02:18.330 growth ensure the right views government
    00:02:21.420 interference in business as a bad thing
    00:02:23.700 and thinks that the market prospers best
    00:02:25.860 when left to its own devices parties on
    00:02:28.590 the Left include labor the greens and
    00:02:30.870 the Democratic Party while those on the
    00:02:33.030 right include the Republicans
    00:02:34.440 conservatives and the UK Independence
    00:02:36.240 Party there is also a center ground
    00:02:38.940 where parties like the Liberal Democrats
    00:02:40.800 lie these parties hold views from both
    00:02:43.140 the left and the right this will be in
    00:02:46.200 said the use of left and right to
    00:02:48.150 describe modern political parties is not
    00:02:50.310 always accurate there are actually many
    00:02:52.590 different political spectrums that can
    00:02:54.300 be used to judge how far left or right
    00:02:56.730 to party years parties often judge
    00:02:59.190 themselves based on other current
    00:03:01.020 political parties in this regard labor
    00:03:03.720 is often seen to be on the left and
    00:03:05.670 conservatives are often seen to be on
    00:03:07.350 the right in actual fact lots of labor
    00:03:10.140 policy actually places it in the center
    00:03:12.300 or right of the political spectrum the
    00:03:14.550 same is true in America where both the
    00:03:16.440 Democrats and Republicans are on the
    00:03:18.210 right of the spectrum in political terms
    00:03:20.040 even if the Democrats are sometimes seen
    00:03:22.530 as having more left-wing values the
    00:03:25.050 political spectrum is also not just
    00:03:27.320 2-dimensional within both the left and
    00:03:29.910 the right there are people who hold
    00:03:31.730 authoritarian and libertarian views
    00:03:34.700 authoritarians wish for greater
    00:03:36.450 government interference and libertarians
    00:03:38.580 want the opposite as previously stated
    00:03:40.830 the Left tend to be authoritarian and
    00:03:43.410 the right tend to be libertarian but
    00:03:45.600 this is not always the case
    00:03:47.040 we are often taught the extremes of
    00:03:49.350 left-wing is communism and the extremes
    00:03:51.570 of right-wing as fascism this is not
    00:03:53.730 strictly true as the extreme
    00:03:55.530 authoritarian nature of fascist
    00:03:57.420 governments combined with left-wing
    00:03:59.250 economic beliefs make it hard to place
    00:04:01.290 on a political scale while it is useful
    00:04:04.440 to have these labels in order to help
    00:04:06.060 the majority of people easily
    00:04:07.410 distinguish between different ideologies
    00:04:09.540 it is also important to remember that
    00:04:11.820 they are not always accurate and looking
    00:04:13.800 at the party's policy for yourself is
    00:04:15.750 often the best way to choose who gets
    00:04:17.640 your vote rather than following a label
    00:04:20.070 blindly
    00:04:21.290 [Music]
    00:04:34.430 you
    """
    result = classify_long_text(transcript)
    print(result)
