class Kiosk:
    def __init__(self):
        self.menu = ['americano', 'latte', 'mocha', 'yuza_tea', 'green_tea', 'choco_latte']
        self.price = [2000, 3000, 3000, 2500, 2500, 3000]
        self.order_menu = []
        self.order_price = []

    # 메뉴 출력 메서드
    def menu_print(self):
        for i in range(len(self.menu)):
            print(i + 1, self.menu[i], ' : ', self.price[i], '원')

    # 주문 메서드
    def menu_select(self):
        print()  # 줄바꿈
        n = int(input("음료 번호를 입력하세요 : "))
        self.price_sum = self.price[n - 1]  # 선택음료의 가격
        print(self.menu[n - 1], ':', self.price[n - 1], '원')

        while n != 0:
            # hot 인지 ice 인지 물어보기
            t = 0  # 기본값 넣기
            while t != 1 and t != 2:  # 1 이나 2를 입력 할때까지 물어보기
                t = int(input("HOT 음료는 1을, ICE 음료는 2를 입력하세요 : "))
                if t == 1:
                    self.temp = "HOT"  # 온도 속성 생성
                elif t == 2:
                    self.temp = "ICE"
                else:
                    print("1과 2 중 하나를 입력하세요.\n")
            print('추가 주문 음료', self.temp, self.menu[n - 1], ':', self.price[n - 1], '원')  # 온도 속성을 추가한 주문 결과 출력

            # 추가 주문 또는 지불
            while n != 0:
                print()
                n = int(input('추가 주문은 음료 번호를,지불은 0을 누르세요:'))  # 추가 주문 1-6 또는 지불 0
                if n > 0 and n < len(self.menu) + 1:  # 1-6번
                    self.order_price.append(self.price[n - 1])  # 가격 리스트에 추가
                    self.order_menu.append(self.temp + ' ' + self.menu[n - 1])  # 주문 리스트에 추가
                    self.price_sum += self.price[n - 1]  # 합계금액

                    print('추가 주문 음료', self.temp, self.menu[n - 1], ' : ', self.price[n - 1], '원\n', '합계 : ',
                          self.price_sum, '원')
                else:
                    if n == 0:  # 지불을 입력하면
                        print("주문이 완료 되었습니다")
                        print(self.order_menu, self.order_price)  # 확인용 리스트 출력해보기
                    else:  # 없는 번호를 입력할때
                        print("없는 메뉴 입니다")

    # 지불
    def pay(self):
        # 합계금액 출력
        print(f"총 합계 금액은 {self.price_sum}원 입니다.")
        # 지불 방법
        self.pay_method = 0  # 기본값
        while self.pay_method != 1 and self.pay_method != 2:  # 1,2번을 입력하지 않으면 계속 해서 반복
            self.pay_method = int(input('현금 결제는 1번,카드 결제는 2번을 입력해 주세요 :'))

            if self.pay_method == 1:
                print("직원을 호출하겠습니다.")
            elif self.pay_method == 2:
                print("IC칩 방향에 맞게 카드를 꽂아주세요.")
            else:
                print("다시 결제를 시도해 주세요.")

    # 주문서 출력
    def table(self):
        # 외곽
        outline1 = '⟝' + '-' * 30 + '⟞'
        outline2 = '|' + ' ' * 31 + '|'
        # 외곽 출력
        print(outline1)
        for i in range(5):
            print(outline2)

        # 주문 내역
        for i in range(len(self.order_menu)):
            print(self.order_menu[i], ' : ', self.order_price[i])
        print('합계 금액 :', self.price_sum)

        # 외곽 출력
        for i in range(5):
            print(outline2)
        print(outline1)