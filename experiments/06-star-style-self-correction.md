# 06. STaR-Style Self-Correction

STaR(Self-Taught Reasoner)의 아이디어를 참고해, wrong reasoning을 다시 학습 데이터로 바꾸는 방향을 검토했습니다.

## Idea

```text
generated reasoning
-> answer extraction
-> checker로 correct/incorrect 분리
-> incorrect reasoning의 failure point 분석
-> corrected trace 또는 rejected branch 구성
```

## Use In This Project

full automated STaR loop를 완성했다고 주장하지 않습니다. 대신 다음 설계로 이어졌습니다.

- correct reasoning은 SFT row 후보
- wrong reasoning은 preference rejected branch 후보
- failure point는 RSP decision point 후보

## Related Files

STaR-style 검토는 `decision_preferences` 설계로 이어졌습니다.
